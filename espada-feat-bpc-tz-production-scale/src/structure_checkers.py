def _is_images_paths_empty(markup: dict) -> None:
    """Checks if the image list is empty."""
    images_paths: list[str] = markup["images"]
    if not images_paths:
        raise ValueError("No images found in bounding box markup.")


def _check_bbox_json_structure(bbox_markup: dict) -> None:
    """Validates JSON structure for bounding box annotations."""

    _is_images_paths_empty(bbox_markup)

    answer: list[dict] = bbox_markup["answer"]
    for i, obj in enumerate(answer):
        if not isinstance(obj, dict):
            raise TypeError(f"Annotation at index {i} must be a dictionary.")

        # Label consistency checking
        if "label" not in obj:
            raise KeyError(f"Missing 'label' in annotation {i}.")

        label: str = obj["label"]

        if not label:
            raise ValueError(f"'label' in annotation {i} must not be empty.")

        # Bbox 2d consistency checking
        if "bbox_2d" not in obj:
            raise KeyError(f"Missing 'bbox_2d' in annotation {i}.")

        bbox: list[int] = obj["bbox_2d"]

        if len(bbox) != 4:
            raise ValueError(f"'bbox_2d' in annotation {i} must have 4 elements.")

        x_min, y_min, x_max, y_max = bbox
        for coord in bbox:
            if not isinstance(coord, int):
                raise TypeError(f"All bbox coordinates in annotation {i} must be numbers.")

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Invalid bbox in annotation {i}: "
                f"(x_min, y_min) must be less than (x_max, y_max)."
            )

        if x_min < 0 or y_min < 0:
            raise ValueError(f"Invalid bbox in annotation {i}: coordinates must be non-negative.")


def _check_pointing_json_structure(pointing_markup: dict) -> None:
    """Validates JSON structure for pointing annotations."""

    _is_images_paths_empty(pointing_markup)

    answer: list[dict] = pointing_markup["answer"]
    for i, obj in enumerate(answer):
        if not isinstance(obj, dict):
            raise TypeError(f"Annotation at index {i} must be a dictionary.")

        # Label consistency checking
        if "label" not in obj:
            raise KeyError(f"Missing 'label' in annotation {i}.")

        label: str = obj["label"]

        if not isinstance(label, str):
            raise TypeError(f"'label' in annotation {i} must be a string.")

        if not label:
            raise ValueError(f"'label' in annotation {i} must not be empty.")

        # Point 2d consistency checking
        if "point_2d" not in obj:
            raise KeyError(f"Missing 'point_2d' in annotation {i}.")

        point: list[int] = obj["point_2d"]

        if not isinstance(point, list):
            raise TypeError(f"'point_2d' in annotation {i} must be a list.")

        if len(point) != 2:
            raise ValueError(f"'point_2d' in annotation {i} must have 2 elements.")

        x, y = point

        for coord in point:
            if not isinstance(coord, int):
                raise TypeError(f"All point coordinates in annotation {i} must be integers.")

        if x < 0 or y < 0:
            raise ValueError(f"Invalid point in annotation {i}: coordinates must be non-negative.")


def _check_captioning_json_structure(captioning_markup: dict) -> None:
    """Validates JSON structure for captioning annotations."""

    _is_images_paths_empty(captioning_markup)

    answer = captioning_markup["answer"]

    if not isinstance(answer, str):
        raise TypeError("'answer' must be a string.")

    if not answer.strip():
        raise ValueError("'answer' must not be empty.")
