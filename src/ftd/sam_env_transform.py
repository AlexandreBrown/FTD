import copy
import torch
import numpy as np
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator


class SamEnvTransform(Transform):
    def __init__(self, device: str, in_key: str, out_key: str, segmenter_model_name: str = "efficientvit-sam-l0", segmenter_weights_path: str = "weights/efficientvit_sam_l0.pt"):
        super().__init__(device)
        self.in_key = in_key
        self.out_key = out_key
        self.segmenter_model = create_efficientvit_sam_model(
            name=segmenter_model_name,
            pretrained=True,
            weight_url=segmenter_weights_path,
        )
        self.segmenter_model = self.segmenter_model.to(device).eval()
        self.segments_predictor = EfficientViTSamAutomaticMaskGenerator(
            model=self.segmenter_model,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.5,
            points_per_side=8,
            points_per_batch=64,
        )
        self.reverse_sort = True
        self.min_area = 100
        self.max_area = 2000
        self.clip_range = [0, 84]
        self.masked_region_num = 9
        self.add_original_frame = True
        self.color_type = "rgb"
        self.image_crop_size = 84

    def mask_filter(self, masks, image, initial_flag=False, overlap_threshold=0.95):
        if len(masks) == 0:
            return []

        masks = sorted(masks, key=(lambda x: x["area"]), reverse=self.reverse_sort)
        filtered_masks = []
        for mask in masks:
            if mask["area"] > self.max_area or mask["area"] < self.min_area:
                continue
            if (
                np.sum(
                    mask["segmentation"][
                        self.clip_range[0] : self.clip_range[1],
                        self.clip_range[0] : self.clip_range[1],
                    ]
                )
                == 0
            ):
                continue

            contain_flag = False
            for previous_mask in filtered_masks:
                if np.sum(
                    mask["segmentation"] * previous_mask["segmentation"]
                ) > overlap_threshold * np.sum(mask["segmentation"]):
                    contain_flag = True
                    break
            if contain_flag:
                continue

            filtered_masks.append(mask)

        if len(filtered_masks) == 0:
            return []

        if len(filtered_masks) > self.masked_region_num:
            filtered_masks = filtered_masks[: self.masked_region_num]
        elif len(filtered_masks) < self.masked_region_num:
            black_mask = copy.deepcopy(masks[0])
            black_mask["segmentation"] = np.zeros_like(black_mask["segmentation"])
            filtered_masks = filtered_masks + [black_mask] * (
                self.masked_region_num - len(filtered_masks)
            )

        if self.add_original_frame:
            white_mask = copy.deepcopy(masks[0])
            white_mask["segmentation"] = np.ones_like(white_mask["segmentation"])
            filtered_masks = filtered_masks + [white_mask]

        return filtered_masks

    def _generate_image_mask(self, image, initial_flag=False, dtype=np.uint8):
        assert image.shape[-1] in [1, 3], "Image can only be gray or rgb"
        masks = self.segments_predictor.generate(image)
        masks = self.mask_filter(masks, image, initial_flag)

        if len(masks) == 0:
            print("No mask detected!")
            if self.color_type == "rgb":
                black_image = np.zeros((3, self.image_crop_size, self.image_crop_size))
                total_images = [image.transpose((2, 0, 1))] + [black_image] * (
                    self.masked_region_num - 1
                )
                if self.add_original_frame:
                    total_images = total_images + [image.transpose((2, 0, 1))]
                total_images = np.concatenate(total_images, axis=0).astype(dtype)
                return total_images
            elif self.color_type == "gray":
                black_image = np.zeros((1, self.image_crop_size, self.image_crop_size))
                gray_image = image.mean(axis=2).astype("uint8")
                total_images = [np.expand_dims(gray_image, axis=0)] + [black_image] * (
                    self.masked_region_num - 1
                )
                if self.add_original_frame:
                    total_images = total_images + [np.expand_dims(gray_image, axis=0)]
                total_images = np.concatenate(total_images).astype(dtype)
                return total_images

        if self.color_type == "rgb":
            total_masks = []
            for item in masks:
                m = np.expand_dims(item["segmentation"], axis=0)
                m = np.expand_dims(
                    np.concatenate([m for _ in range(3)], axis=0), axis=0
                )
                total_masks.append(m)
            total_masks = np.concatenate(total_masks, axis=0)
            total_images = np.concatenate(
                [
                    np.expand_dims(image.transpose((2, 0, 1)), axis=0)
                    for _ in range(len(masks))
                ]
            )
            masked_images = (total_masks * total_images).reshape(
                len(masks) * 3, self.image_crop_size, self.image_crop_size
            )

            zero_position_sum = np.sum(
                masked_images.reshape(
                    (3, 10, self.image_crop_size, self.image_crop_size)
                )[:, 0, :, :],
                axis=(0, 1, 2),
            )
            if zero_position_sum == 0:
                print("zero position is black")
                masked_images[:3, :, :] = image.transpose((2, 0, 1))
        elif self.color_type == "gray":
            total_masks = []
            for item in masks:
                m = np.expand_dims(item["segmentation"], axis=0)
                total_masks.append(m)
            total_masks = np.concatenate(total_masks, axis=0)
            gray_image = image.mean(axis=2).astype("uint8")
            total_images = np.concatenate(
                [np.expand_dims(gray_image, axis=0) for _ in range(len(masks))]
            )
            masked_images = total_masks * total_images

        return masked_images.astype(dtype)

    def reset(self, mdp_data: MdpData) -> MdpData:
        images = mdp_data.data[self.in_key].squeeze(1)  # (b,c,h,w)
        images = images.permute(0, 2, 3, 1)  # (b,h,w,c)
        images = images.cpu().numpy()
        masked_images = []
        for img in images:
            pred = self._generate_image_mask(img, initial_flag=True)
            masked_images.append(torch.from_numpy(pred).reshape(10, 3, 84, 84))
        mdp_data.data[self.out_key] = torch.stack(masked_images, dim=0).unsqueeze(1)  # (b,1,seg,c,h,w)

        return mdp_data

    def step(self, mdp_data: MdpData) -> MdpData:
        images = mdp_data.data[self.in_key].squeeze(1)  # (b,c,h,w)
        images = images.permute(0, 2, 3, 1)  # (b,h,w,c)
        images = images.cpu().numpy()
        masked_images = []
        for img in images:
            pred = self._generate_image_mask(img, initial_flag=False)
            masked_images.append(torch.from_numpy(pred).reshape(10, 3, 84, 84))
        mdp_data.data[self.out_key] = torch.stack(masked_images, dim=0).unsqueeze(1)  # (b,1,seg,c,h,w)

        return mdp_data
