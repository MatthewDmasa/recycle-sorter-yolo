from dataclasses import dataclass


@dataclass
class SortingResult:
    """
    Represents the sorting recommendation for one detected object.
    """
    detected_class: str
    sort_category: str
    sort_message: str


# Map model class names to broader sorting categories
SORTING_RULES = {
    "plastic": {
        "category": "Plastic",
        "message": "Sort into plastic recycling."
    },
    "detergent": {
        "category": "Plastic",
        "message": "Sort into plastic recycling. Rinse if needed."
    },
    "can": {
        "category": "Metal",
        "message": "Sort into metal recycling."
    },
    "cardboard": {
        "category": "Cardboard",
        "message": "Sort into cardboard/paper recycling."
    },
    "glass": {
        "category": "Glass",
        "message": "Sort into glass recycling."
    },
    "canister": {
        "category": "Plastic",
        "message": "Sort based on local rules. Default: plastic/container recycling."
    }
}


def get_sorting_recommendation(detected_class: str) -> SortingResult:
    """
    Convert a model class name into a broader sorting recommendation.

    Args:
        detected_class: Predicted YOLO class label

    Returns:
        SortingResult: Structured sorting recommendation
    """
    detected_class = detected_class.strip().lower()

    rule = SORTING_RULES.get(
        detected_class,
        {
            "category": "Unknown",
            "message": "No sorting recommendation available."
        }
    )

    return SortingResult(
        detected_class=detected_class,
        sort_category=rule["category"],
        sort_message=rule["message"]
    )


if __name__ == "__main__":
    example = get_sorting_recommendation("detergent")
    print(example)