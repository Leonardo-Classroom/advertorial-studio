"""Staff-only user administration.

Django's own admin can do all of this, but it lives at a different URL with a
different look, and the point of putting it here is that whoever runs the
system manages people in the same place they manage everything else.

Deactivation rather than deletion is the only removal offered: a deleted user
would take the attribution of their briefs and drafts with them, and the work
usually outlives the account.
"""
from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AdminPasswordChangeForm, UserCreationForm
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render

from accounts.decorators import staff_required

User = get_user_model()


@staff_required
def user_list(request):
    users = (User.objects.annotate(n_briefs=Count("briefs", distinct=True),
                                   n_runs=Count("runs", distinct=True))
             .order_by("-is_staff", "username"))
    return render(request, "accounts/list.html", {"section": "users", "users": users})


@staff_required
def user_create(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = (request.POST.get("email") or "").strip()
            user.first_name = (request.POST.get("first_name") or "").strip()
            user.is_staff = request.POST.get("is_staff") == "on"
            user.save()
            messages.success(
                request,
                f"已建立帳號「{user.username}」"
                + ("，具管理者權限。" if user.is_staff else "，只能使用前台。"),
            )
            return redirect("accounts:list")
        messages.error(request, "建立失敗，請看下方欄位說明。")
    else:
        form = UserCreationForm()
    return render(request, "accounts/form.html",
                  {"section": "users", "form": form, "creating": True})


@staff_required
def user_edit(request, pk):
    user = get_object_or_404(User, pk=pk)
    if request.method == "POST":
        user.email = (request.POST.get("email") or "").strip()
        user.first_name = (request.POST.get("first_name") or "").strip()

        wants_staff = request.POST.get("is_staff") == "on"
        wants_active = request.POST.get("is_active") == "on"

        # Guard against locking everyone out of the admin side.
        if user == request.user and (not wants_staff or not wants_active):
            messages.error(request, "不能移除自己的管理權限或停用自己的帳號。")
            return redirect("accounts:edit", pk=pk)
        if not wants_staff and user.is_staff and User.objects.filter(
                is_staff=True, is_active=True).count() <= 1:
            messages.error(request, "這是最後一個管理者帳號，不能取消其權限。")
            return redirect("accounts:edit", pk=pk)

        user.is_staff = wants_staff
        user.is_active = wants_active
        user.save(update_fields=["email", "first_name", "is_staff", "is_active"])
        messages.success(request, "已更新。")
        return redirect("accounts:list")

    return render(request, "accounts/form.html",
                  {"section": "users", "edit_user": user, "creating": False})


@staff_required
def user_reset_password(request, pk):
    user = get_object_or_404(User, pk=pk)
    if request.method == "POST":
        form = AdminPasswordChangeForm(user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f"已重設「{user.username}」的密碼。")
            return redirect("accounts:list")
        messages.error(request, "重設失敗，請看下方欄位說明。")
    else:
        form = AdminPasswordChangeForm(user)
    return render(request, "accounts/password.html",
                  {"section": "users", "form": form, "edit_user": user})
