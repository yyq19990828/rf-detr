# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import shutil
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.yolo import CocoLikeAPI, ConvertYolo, YoloDetection, _MockSvDataset, is_valid_yolo_dataset


class TestCocoLikeAPI:
    """Tests for the CocoLikeAPI class."""

    @pytest.fixture
    def coco_api(self):
        """Fixture to create a test instance of CocoLikeAPI."""
        mock = _MockSvDataset()
        sizes = {"img_0.jpg": (100, 100), "img_1.jpg": (100, 100)}
        return CocoLikeAPI(mock.classes, mock, sizes)

    def test_initialization(self, coco_api):
        """Test that the API initializes correctly."""
        assert coco_api is not None
        assert hasattr(coco_api, "dataset")
        assert hasattr(coco_api, "imgs")
        assert hasattr(coco_api, "anns")
        assert hasattr(coco_api, "cats")
        assert hasattr(coco_api, "imgToAnns")
        assert hasattr(coco_api, "catToImgs")

    def test_dataset_structure(self, coco_api):
        """Test the structure of the COCO dataset."""
        assert "info" in coco_api.dataset
        assert "images" in coco_api.dataset
        assert "annotations" in coco_api.dataset
        assert "categories" in coco_api.dataset

    @pytest.mark.parametrize(
        "dataset_part, expected_count",
        [
            ("images", 2),
            ("categories", 2),
            ("annotations", 2),
        ],
    )
    def test_dataset_counts(self, coco_api, dataset_part, expected_count):
        """Test the number of images, categories, and annotations in the dataset."""
        assert len(coco_api.dataset[dataset_part]) == expected_count

    @pytest.mark.parametrize(
        "img_ids, expected_ids",
        [
            (None, [0, 1]),
            ([0], [0]),
            ([1], [1]),
            ([0, 1], [0, 1]),
        ],
    )
    def test_get_img_ids_by_img_ids(self, coco_api, img_ids, expected_ids):
        """Test getImgIds method with various image ID filters."""
        result = coco_api.getImgIds(imgIds=img_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "cat_ids, expected_img_ids",
        [
            (None, [0, 1]),
            ([0], [0]),
            ([1], [1]),
            ([0, 1], [0, 1]),
        ],
    )
    def test_get_img_ids_by_cat_ids(self, coco_api, cat_ids, expected_img_ids):
        """Test getImgIds method with various category ID filters."""
        result = coco_api.getImgIds(catIds=cat_ids)
        assert sorted(result) == sorted(expected_img_ids)

    @pytest.mark.parametrize(
        "cat_names, expected_ids",
        [
            (None, [0, 1]),
            (["cat"], [0]),
            (["dog"], [1]),
            (["cat", "dog"], [0, 1]),
        ],
    )
    def test_get_cat_ids_by_names(self, coco_api, cat_names, expected_ids):
        """Test getCatIds method with various category name filters."""
        result = coco_api.getCatIds(catNms=cat_names)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "cat_ids, expected_ids",
        [
            (None, [0, 1]),
            ([0], [0]),
            ([1], [1]),
            ([0, 1], [0, 1]),
        ],
    )
    def test_get_cat_ids_by_ids(self, coco_api, cat_ids, expected_ids):
        """Test getCatIds method with various category ID filters."""
        result = coco_api.getCatIds(catIds=cat_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "img_ids, cat_ids, expected_ids",
        [
            (None, None, [0, 1]),
            ([0], None, [0]),
            (None, [1], [1]),
            ([0], [0], [0]),
        ],
    )
    def test_get_ann_ids(self, coco_api, img_ids, cat_ids, expected_ids):
        """Test getAnnIds method with various filter conditions."""
        result = coco_api.getAnnIds(imgIds=img_ids, catIds=cat_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "ann_ids, expected_length",
        [
            ([0], 1),
            ([1], 1),
            ([0, 1], 2),
        ],
    )
    def test_load_anns(self, coco_api, ann_ids, expected_length):
        """Test loadAnns method with various annotation IDs."""
        result = coco_api.loadAnns(ann_ids)
        assert len(result) == expected_length
        assert all(ann["id"] in ann_ids for ann in result)

    @pytest.mark.parametrize(
        "cat_ids, expected_length",
        [
            ([0], 1),
            ([1], 1),
            ([0, 1], 2),
            (None, 2),
        ],
    )
    def test_load_cats(self, coco_api, cat_ids, expected_length):
        """Test loadCats method with various category IDs."""
        result = coco_api.loadCats(cat_ids)
        assert len(result) == expected_length
        if cat_ids is not None:
            assert all(cat["id"] in cat_ids for cat in result)

    @pytest.mark.parametrize(
        "img_ids, expected_length",
        [
            ([0], 1),
            ([1], 1),
            ([0, 1], 2),
        ],
    )
    def test_load_imgs(self, coco_api, img_ids, expected_length):
        """Test loadImgs method with various image IDs."""
        result = coco_api.loadImgs(img_ids)
        assert len(result) == expected_length
        assert all(img["id"] in img_ids for img in result)

    def test_img_to_anns(self, coco_api):
        """Test the imgToAnns index."""
        assert len(coco_api.imgToAnns[0]) == 1
        assert len(coco_api.imgToAnns[1]) == 1
        assert coco_api.imgToAnns[0][0]["id"] == 0
        assert coco_api.imgToAnns[1][0]["id"] == 1

    def test_cat_to_imgs(self, coco_api):
        """Test the catToImgs index."""
        assert len(coco_api.catToImgs[0]) == 1
        assert len(coco_api.catToImgs[1]) == 1
        assert 0 in coco_api.catToImgs[0]
        assert 1 in coco_api.catToImgs[1]

    @pytest.mark.parametrize("ann_id", [0, 1])
    def test_annotation_format(self, coco_api, ann_id):
        """Test that annotations are in the correct format."""
        ann = coco_api.loadAnns([ann_id])[0]

        # Check required fields
        required_fields = ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]
        for field in required_fields:
            assert field in ann, f"Annotation missing required field: {field}"

        # Check bbox format
        assert len(ann["bbox"]) == 4, "BBox must have 4 coordinates"
        assert all(isinstance(x, (int, float)) for x in ann["bbox"]), "BBox coordinates must be numeric"

        # Check area
        assert isinstance(ann["area"], (int, float)), "Area must be numeric"
        assert ann["area"] > 0, "Area must be positive"

        # Check iscrowd
        assert ann["iscrowd"] in [0, 1], "iscrowd must be 0 or 1"

    @pytest.mark.parametrize("cat_id", [0, 1])
    def test_category_format(self, coco_api, cat_id):
        """Test that categories are in the correct format."""
        cat = coco_api.loadCats([cat_id])[0]

        # Check required fields
        required_fields = ["id", "name", "supercategory"]
        for field in required_fields:
            assert field in cat, f"Category missing required field: {field}"

        # Check field types
        assert isinstance(cat["id"], int), "Category ID must be an integer"
        assert isinstance(cat["name"], str), "Category name must be a string"
        assert isinstance(cat["supercategory"], str), "Supercategory must be a string"

    @pytest.mark.parametrize("img_id", [0, 1])
    def test_image_format(self, coco_api, img_id):
        """Test that images are in the correct format."""
        img = coco_api.loadImgs([img_id])[0]

        # Check required fields
        required_fields = ["id", "file_name", "width", "height"]
        for field in required_fields:
            assert field in img, f"Image missing required field: {field}"

        # Check field types
        assert isinstance(img["id"], int), "Image ID must be an integer"
        assert isinstance(img["file_name"], str), "File name must be a string"
        assert isinstance(img["width"], int), "Width must be an integer"
        assert isinstance(img["height"], int), "Height must be an integer"

    def test_empty_annotations(self):
        """Test handling of images with no annotations."""

        class EmptyMockDataset(_MockSvDataset):
            def __init__(self):
                super().__init__()
                self.image_paths = ["img_0.jpg"]
                self.annotations = {
                    "img_0.jpg": sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([])),
                }

        sizes = {"img_0.jpg": (100, 100)}
        api = CocoLikeAPI(["cat"], EmptyMockDataset(), sizes)
        assert len(api.dataset["annotations"]) == 0
        assert len(api.getAnnIds()) == 0

    def test_images_with_multiple_annotations(self):
        """Test handling of images with multiple annotations per image."""

        class MultiAnnotationMockDataset(_MockSvDataset):
            def __init__(self):
                super().__init__()
                self.image_paths = ["img_0.jpg", "img_1.jpg"]
                self.annotations = {
                    "img_0.jpg": sv.Detections(
                        xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]), class_id=np.array([0, 1])
                    ),
                    "img_1.jpg": sv.Detections(xyxy=np.array([[15, 25, 35, 45]]), class_id=np.array([0])),
                }

        sizes = {"img_0.jpg": (100, 100), "img_1.jpg": (100, 100)}
        api = CocoLikeAPI(["cat", "dog"], MultiAnnotationMockDataset(), sizes)

        # Verify 3 annotations in total
        assert len(api.dataset["annotations"]) == 3

        # Verify annotations per image
        assert len(api.imgToAnns[0]) == 2
        assert len(api.imgToAnns[1]) == 1

        # Verify image IDs per category
        assert 0 in api.catToImgs[0]
        assert 1 in api.catToImgs[0]
        assert 0 in api.catToImgs[1]


class TestBuildRoboflowFromYoloAugConfig:
    """Regression tests for #769: aug_config forwarded to transform builders."""

    def _make_args(self, square_resize_div_64: bool, aug_config=None) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            dataset_dir="/fake/dataset",
            square_resize_div_64=square_resize_div_64,
            aug_config=aug_config,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=None,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
        )

    @pytest.mark.parametrize(
        "square_resize_div_64,transform_fn,aug_config",
        [
            pytest.param(
                True,
                "make_coco_transforms_square_div_64",
                {"HorizontalFlip": {"p": 0.5}},
                id="square_div_64_with_config",
            ),
            pytest.param(False, "make_coco_transforms", {"HorizontalFlip": {"p": 0.5}}, id="standard_with_config"),
            pytest.param(True, "make_coco_transforms_square_div_64", None, id="square_div_64_none"),
            pytest.param(False, "make_coco_transforms", None, id="standard_none"),
        ],
    )
    def test_aug_config_forwarded_to_transform(
        self, square_resize_div_64: bool, transform_fn: str, aug_config: object
    ) -> None:
        """Regression test for #769: aug_config is forwarded to transform builders for all code paths."""
        args = self._make_args(square_resize_div_64=square_resize_div_64, aug_config=aug_config)

        with (
            patch("rfdetr.datasets.yolo.Path") as mock_path,
            patch(f"rfdetr.datasets.yolo.{transform_fn}") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
        ):
            mock_path.return_value.exists.return_value = True
            mock_transform.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()

            from rfdetr.datasets.yolo import build_roboflow_from_yolo

            build_roboflow_from_yolo("train", args, resolution=640)

        _, kwargs = mock_transform.call_args
        assert kwargs.get("aug_config") == aug_config, (
            f"{transform_fn} was not called with aug_config={aug_config!r}; got {kwargs}"
        )

    def test_data_yml_selected_when_data_yaml_missing(self, tmp_path: Path) -> None:
        """Regression test: build_roboflow_from_yolo picks data.yml when data.yaml is not present."""
        (tmp_path / "data.yml").touch()
        args = self._make_args(square_resize_div_64=False, aug_config=None)
        args.dataset_dir = str(tmp_path)

        with (
            patch("rfdetr.datasets.yolo.make_coco_transforms") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
        ):
            mock_transform.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()

            from rfdetr.datasets.yolo import build_roboflow_from_yolo

            build_roboflow_from_yolo("train", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        assert kwargs["data_file"] == str(tmp_path / "data.yml")


class TestIsValidYoloDataset:
    """Tests for the is_valid_yolo_dataset function."""

    def _create_valid_yolo_dataset(self, tmp_path: Path, yaml_filename: str) -> str:
        """Create a minimal valid YOLO dataset directory structure."""
        (tmp_path / yaml_filename).touch()
        for split in ["train", "valid"]:
            for subdir in ["images", "labels"]:
                (tmp_path / split / subdir).mkdir(parents=True)
        return str(tmp_path)

    @pytest.mark.parametrize(
        "yaml_filename",
        [
            pytest.param("data.yaml", id="data_yaml"),
            pytest.param("data.yml", id="data_yml"),
        ],
    )
    def test_valid_dataset_with_yaml_variants(self, tmp_path: Path, yaml_filename: str) -> None:
        """Regression test: both data.yaml and data.yml are accepted as valid YOLO datasets."""
        dataset_dir = self._create_valid_yolo_dataset(tmp_path, yaml_filename)
        assert is_valid_yolo_dataset(dataset_dir) is True

    def test_invalid_dataset_missing_yaml(self, tmp_path: Path) -> None:
        """Dataset without any YAML file should be invalid."""
        for split in ["train", "valid"]:
            for subdir in ["images", "labels"]:
                (tmp_path / split / subdir).mkdir(parents=True)
        assert is_valid_yolo_dataset(str(tmp_path)) is False

    def test_invalid_dataset_missing_split_dirs(self, tmp_path: Path) -> None:
        """Dataset without required split directories should be invalid."""
        (tmp_path / "data.yaml").touch()
        assert is_valid_yolo_dataset(str(tmp_path)) is False


class TestLoadYoloAnnotationsCached:
    """Tests for load_yolo_annotations_cached with v5 cache optimizations."""

    @staticmethod
    def _create_dataset(tmp_path: Path, num_images: int = 3) -> tuple[Path, Path, Path]:
        """Create a minimal YOLO dataset on disk for cache testing.

        Returns:
            Tuple of (images_dir, labels_dir, data_yaml_path).
        """
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names:\n  0: cat\n  1: dog\n")

        for i in range(num_images):
            # Create a small valid PNG image
            img = Image.new("RGB", (80 + i, 60 + i), color=(i * 40, 100, 200))
            img.save(images_dir / f"img_{i:03d}.png")

            # Create a matching YOLO label
            labels_dir.joinpath(f"img_{i:03d}.txt").write_text(f"0 0.5 0.5 0.4 0.3\n{1} 0.2 0.2 0.1 0.1\n")

        return images_dir, labels_dir, data_yaml

    def test_cache_written_and_loaded(self, tmp_path: Path) -> None:
        """First call writes .cache; second call loads from it with image_sizes."""
        from rfdetr.datasets.yolo import load_yolo_annotations_cached

        images_dir, labels_dir, data_yaml = self._create_dataset(tmp_path)
        cache_file = labels_dir / ".cache"

        # First call – full scan, writes cache
        ds1, sizes1 = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))
        assert cache_file.exists()
        assert len(ds1) == 3
        assert len(sizes1) == 3

        # Second call – should load from cache
        ds2, sizes2 = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))
        assert len(ds2) == 3
        assert sizes2 == sizes1

    def test_image_sizes_are_real_dimensions(self, tmp_path: Path) -> None:
        """image_sizes should contain real (width, height) from image headers."""
        from rfdetr.datasets.yolo import load_yolo_annotations_cached

        images_dir, labels_dir, data_yaml = self._create_dataset(tmp_path, num_images=2)

        _, sizes = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))

        for path, (w, h) in sizes.items():
            assert w > 0 and h > 0, f"Expected positive dims, got ({w}, {h})"
            # Verify against actual image
            with Image.open(path) as img:
                assert (w, h) == img.size

    def test_corrupt_image_excluded(self, tmp_path: Path) -> None:
        """Corrupt images should be excluded via Image.verify() during scan."""
        from rfdetr.datasets.yolo import load_yolo_annotations_cached

        images_dir, labels_dir, data_yaml = self._create_dataset(tmp_path, num_images=2)

        # Corrupt one image
        corrupt_img = images_dir / "img_000.png"
        corrupt_img.write_bytes(b"not a valid image at all")

        ds, sizes = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))

        # Only 1 valid image should remain
        assert len(ds) == 1
        assert len(sizes) == 1
        assert str(corrupt_img) not in sizes

    def test_cache_stale_after_new_image(self, tmp_path: Path) -> None:
        """Adding a new image should invalidate the cache and trigger rescan."""
        from rfdetr.datasets.yolo import load_yolo_annotations_cached

        images_dir, labels_dir, data_yaml = self._create_dataset(tmp_path, num_images=2)

        # Build cache with 2 images
        ds1, _ = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))
        assert len(ds1) == 2

        # Add a third image + label
        img = Image.new("RGB", (100, 100))
        img.save(images_dir / "img_new.png")
        (labels_dir / "img_new.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        # Should rescan and find 3 images
        ds2, sizes2 = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))
        assert len(ds2) == 3
        assert len(sizes2) == 3

    def test_cache_stale_after_dataset_is_copied_to_new_root(self, tmp_path: Path) -> None:
        """A copied dataset should not reuse cached absolute image paths from the source root."""
        from rfdetr.datasets.yolo import load_yolo_annotations_cached

        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()

        images_dir, labels_dir, data_yaml = self._create_dataset(source_root, num_images=2)

        ds1, _ = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))
        assert all(path.startswith(str(images_dir)) for path in ds1.image_paths)

        shutil.copytree(source_root, target_root)

        target_images_dir = target_root / "images"
        target_labels_dir = target_root / "labels"
        target_data_yaml = target_root / "data.yaml"

        ds2, sizes2 = load_yolo_annotations_cached(
            str(target_images_dir),
            str(target_labels_dir),
            str(target_data_yaml),
        )

        assert len(ds2) == 2
        assert len(sizes2) == 2
        assert all(path.startswith(str(target_images_dir)) for path in ds2.image_paths)

    def test_cache_contains_image_sizes_key(self, tmp_path: Path) -> None:
        """The serialized cache dict must contain an 'image_sizes' key."""
        import pickle

        from rfdetr.datasets.yolo import load_yolo_annotations_cached

        images_dir, labels_dir, data_yaml = self._create_dataset(tmp_path, num_images=1)
        load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(data_yaml))

        cache_data = pickle.loads((labels_dir / ".cache").read_bytes())
        assert "image_sizes" in cache_data
        assert len(cache_data["image_sizes"]) == 1

    def test_missing_label_gets_empty_detections(self, tmp_path: Path) -> None:
        """Images without a matching label file should get empty detections."""
        from rfdetr.datasets.yolo import load_yolo_annotations_cached

        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        (tmp_path / "data.yaml").write_text("names:\n  0: thing\n")

        # Create image but no label
        img = Image.new("RGB", (50, 50))
        img.save(images_dir / "lonely.png")

        ds, sizes = load_yolo_annotations_cached(str(images_dir), str(labels_dir), str(tmp_path / "data.yaml"))
        assert len(ds) == 1
        assert len(sizes) == 1
        path = ds.image_paths[0]
        assert len(ds.annotations[path]) == 0


class TestMultiDirYoloDataset:
    @staticmethod
    def _create_split_dataset(root: Path, stem: str) -> None:
        """Create a minimal YOLO dataset root with train/valid splits."""
        root.mkdir()
        (root / "data.yaml").write_text("names:\n  0: car\n")

        for split in ("train", "valid"):
            images_dir = root / split / "images"
            labels_dir = root / split / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            image_path = images_dir / f"{stem}_{split}.png"
            Image.new("RGB", (64, 48), color=(120, 80, 40)).save(image_path)
            (labels_dir / f"{stem}_{split}.txt").write_text("0 0.5 0.5 0.4 0.3\n")

    def test_build_dataset_offsets_image_ids_across_multiple_roots(self, tmp_path: Path) -> None:
        """Multi-root YOLO datasets should expose globally unique image IDs for evaluation."""
        root_a = tmp_path / "dataset_a"
        root_b = tmp_path / "dataset_b"
        self._create_split_dataset(root_a, "a")
        self._create_split_dataset(root_b, "b")

        args = types.SimpleNamespace(
            dataset_file="yolo",
            dataset_dir=[str(root_a), str(root_b)],
            square_resize_div_64=False,
            aug_config=None,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=2,
        )

        dataset = build_dataset("val", args, resolution=640)

        _, target0 = dataset[0]
        _, target1 = dataset[1]

        image_id0 = int(target0["image_id"].reshape(-1)[0].item())
        image_id1 = int(target1["image_id"].reshape(-1)[0].item())

        assert image_id0 != image_id1

        coco_api = get_coco_api_from_dataset(dataset)
        assert coco_api is not None
        assert sorted(coco_api.getImgIds()) == sorted([image_id0, image_id1])


class TestConvertYoloRobustness:
    def test_convert_yolo_filters_non_finite_boxes(self) -> None:
        converter = ConvertYolo(include_masks=False, normalized_coords=False)
        image = Image.new("RGB", (100, 100))
        detections = sv.Detections(
            xyxy=np.array([[10.0, 10.0, 30.0, 30.0], [5.0, 5.0, np.nan, 20.0]], dtype=np.float32),
            class_id=np.array([0, 1], dtype=np.int64),
        )

        _, target = converter(image, {"image_id": 0, "detections": detections})

        assert target["boxes"].shape == (1, 4)
        assert target["labels"].tolist() == [0]


class TestYoloDetectionRobustness:
    def test_getitem_retries_on_invalid_zero_sized_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = YoloDetection.__new__(YoloDetection)
        dataset.ids = [10, 11]
        dataset._transforms = None
        dataset.prepare = lambda img, target: (img, target)

        class _MockSvDataset:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int):
                if idx == 0:
                    bad_image = np.zeros((0, 100, 3), dtype=np.uint8)
                    return "bad.jpg", bad_image, sv.Detections.empty()
                good_image = np.zeros((100, 100, 3), dtype=np.uint8)
                return "good.jpg", good_image, sv.Detections.empty()

        dataset.sv_dataset = _MockSvDataset()
        monkeypatch.setattr("random.randint", lambda _a, _b: 1)

        image, target = dataset[0]

        assert image.size == (100, 100)
        assert target["image_id"] == 11
