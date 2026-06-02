"""
src/ui/family_window.py — Cửa sổ quản lý thành viên gia đình.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from ui.theme import T, make_btn
from schemas import AddFamilyMemberPayload


class FamilyManagementWindow(tk.Toplevel):
    """Cửa sổ quản lý thành viên gia đình (thêm / xóa)."""

    def __init__(self, parent_app):
        super().__init__(parent_app.root)
        self._app = parent_app
        self.title("Quản lý thành viên gia đình")
        self.configure(bg=T["bg"])
        self.geometry("480x520")
        self.resizable(False, False)
        self._build()
        self._refresh()

    def _build(self):
        tk.Label(self, text="THÀNH VIÊN GIA ĐÌNH",
                 font=("Courier New", 13, "bold"), fg=T["accent"], bg=T["bg"]
                 ).pack(pady=(14, 6))

        # List frame
        lf = tk.Frame(self, bg=T["card"],
                      highlightbackground=T["border"], highlightthickness=1)
        lf.pack(fill="both", expand=True, padx=16, pady=6)

        cols = ("Tên", "Vai trò", "Bệnh nhân", "Mẫu", "ID")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", height=14)
        col_widths = {"Tên": 110, "Vai trò": 80, "Bệnh nhân": 70, "Mẫu": 50, "ID": 70}
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=col_widths.get(c, 80), anchor="center")
        self._tree.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(lf, command=self._tree.yview)
        sb.pack(side="right", fill="y")
        self._tree.configure(yscrollcommand=sb.set)

        # Buttons
        bf = tk.Frame(self, bg=T["bg"])
        bf.pack(fill="x", padx=16, pady=10)
        make_btn(bf, "＋ Thêm thành viên",  self._add,    bg=T["ok"],    fg="#000").pack(side="left", padx=4)
        make_btn(bf, "✕ Xóa đã chọn",       self._delete, bg=T["danger"], fg="#fff").pack(side="left", padx=4)
        make_btn(bf, "↺ Làm mới",            self._refresh, fg=T["text"]).pack(side="right", padx=4)

        # Status
        self._status = tk.Label(self, text="", font=("Courier New", 8),
                                fg=T["sub"], bg=T["bg"])
        self._status.pack(pady=(0, 8))

    def _refresh(self):
        for row in self._tree.get_children():
            self._tree.delete(row)
        members = []
        if self._app._worker and self._app._worker._face_engine:
            members = self._app._worker._face_engine.list_members()
        for m in members:
            patient_mark = "✓" if m.get("is_patient") else ""
            self._tree.insert("", "end", values=(
                m["name"], m["role"], patient_mark,
                m["sample_count"], m["person_id"],
            ))
        self._status.config(text=f"{len(members)} thành viên trong database")

    def _add(self):
        frame = self._app._last_frame_bgr
        if frame is None:
            messagebox.showwarning("Chưa có camera",
                                   "Bấm BẮT ĐẦU trước, sau đó thêm thành viên.", parent=self)
            return

        fe = self._app._worker._face_engine if self._app._worker else None
        if fe is None:
            messagebox.showwarning("Nhận diện khuôn mặt chưa bật",
                                   "Bật checkbox 'Nhận diện khuôn mặt' trước.", parent=self)
            return

        name = simpledialog.askstring("Tên thành viên", "Nhập tên:", parent=self)
        if not name:
            return
        role = simpledialog.askstring("Vai trò", "Vai trò (family / caregiver):",
                                      initialvalue="family", parent=self) or "family"
        is_patient = messagebox.askyesno(
            "Bệnh nhân",
            f"'{name.strip()}' có phải bệnh nhân cần theo dõi tư thế không?\n\n"
            "• Có  → hệ thống sẽ gửi thông báo trạng thái (đứng/ngồi/nằm/đi) về mobile\n"
            "• Không → chỉ nhận diện danh tính bình thường",
            parent=self,
        )

        pid = fe.enroll(frame, name=name.strip(), role=role.strip(), is_patient=is_patient)
        if pid is None:
            messagebox.showerror("Không phát hiện khuôn mặt",
                                 "Không thấy khuôn mặt trong frame hiện tại.\n"
                                 "Hãy đứng trước camera rồi thử lại.", parent=self)
            return

        # Đồng bộ vào FamilyManager (in-memory dict) nếu đang active
        fm = self._app._family_manager
        if fm is not None:
            enc = fe.db.members[pid]["encodings"][-1]
            fm.add_encoding(pid, name.strip(), is_patient, enc)

        # Sync metadata lên backend (không bắt buộc — bỏ qua nếu offline)
        payload = AddFamilyMemberPayload(
            person_id=pid, name=name.strip(), role=role.strip(), is_patient=is_patient)
        ok = self._app._backend.add_family_member_sync(payload)
        suffix = "" if ok else " (backend offline — lưu local)"
        patient_note = " [BỆNH NHÂN]" if is_patient else ""
        messagebox.showinfo("Thành công",
                            f"Đã thêm '{name}'{patient_note} (ID: {pid}){suffix}", parent=self)
        self._refresh()

    def _delete(self):
        sel = self._tree.selection()
        if not sel:
            return
        values = self._tree.item(sel[0], "values")
        pid, name = values[4], values[0]
        if not messagebox.askyesno("Xác nhận", f"Xóa '{name}' khỏi database?", parent=self):
            return

        fe = self._app._worker._face_engine if self._app._worker else None
        if fe:
            fe.remove_member(pid)
        if self._app._family_manager:
            self._app._family_manager.remove(pid)
        self._app._backend.remove_family_member_sync(pid)
        self._refresh()
