from django import template

register = template.Library()


@register.filter
def approved_count(images) -> int:
    """How many of these `BriefImage`s are currently ticked as usable.

    For the file/slide group headings in `_image_review.html` — a group's
    total tells you how many pictures are there, this tells you how many of
    them the operator has actually approved so far.
    """
    return sum(1 for image in images if image.approved)
