"""Access control for the staff-only tooling under /manage/.

Django ships `staff_member_required`, but it sends *every* failure to the admin
login page — including someone who is already signed in and simply lacks
permission. Being asked to log in again when you are logged in reads as a bug,
so the two cases are separated: sign in, or you cannot come in.
"""
from __future__ import annotations

from functools import wraps

from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse


def staff_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"{reverse('portal:login')}?next={request.get_full_path()}")
        if not request.user.is_staff:
            messages.error(request, "這個頁面僅限管理者使用。")
            return redirect("portal:home")
        return view(request, *args, **kwargs)

    return wrapper
