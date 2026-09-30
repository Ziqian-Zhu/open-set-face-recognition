"""External camera discovery and lifecycle management."""

from __future__ import annotations

import platform
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import CameraConfig


@dataclass(frozen=True)
class CameraDevice:
    """A camera index discovered through OpenCV."""

    index: int
    label: str
    unique_id: str = ""
    built_in: bool = False


class CameraManager:
    """Own exactly one ``VideoCapture`` and release it predictably."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self.current_index: int | None = None

    @staticmethod
    def _backend() -> int:
        """Use the native macOS backend when available."""

        return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

    def discover(self) -> list[CameraDevice]:
        """Probe a small configured index range and return readable devices."""

        if platform.system() == "Darwin":
            from .macos_devices import list_video_devices
            devices = []
            for item in list_video_devices():
                kind, name = item["device_type"], item["name"]
                if not item["connected"] or any(word in (kind+name).lower()
                        for word in ("continuity", "deskview", "iphone", "连续互通", "桌上视角")):
                    continue
                built_in = "BuiltIn" in kind
                label = f"{'电脑内置' if built_in else '本机设备'} · {name} [{item['index']}]"
                devices.append(CameraDevice(item["index"], label, item["unique_id"], built_in))
            return sorted(devices, key=lambda device: not device.built_in)
        devices: list[CameraDevice] = []
        for index in range(self.config.scan_count):
            capture = cv2.VideoCapture(index, self._backend())
            available = capture.isOpened()
            if available:
                available, _ = capture.read()
            capture.release()
            if available:
                kind = "默认摄像头" if index == self.config.preferred_index else "摄像头"
                devices.append(CameraDevice(index=index, label=f"{kind} {index}"))
        return devices

    def open(self, index: int, *, unique_id: str = "") -> None:
        """Open a selected device and apply requested capture properties."""

        self.release()
        if platform.system() == "Darwin":
            devices = self.discover()
            selected = next((device for device in devices if
                             (device.unique_id == unique_id if unique_id else device.index == index)), None)
            if selected is None:
                raise RuntimeError("所选电脑摄像头已不可用，请刷新设备；不会自动切换到手机相机")
            index = selected.index
        capture = cv2.VideoCapture(index, self._backend())
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开摄像头 {index}，请检查连接或系统相机权限")
        # Native defaults avoid unsupported mode negotiation during macOS startup.
        if platform.system() != "Darwin":
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._capture = capture
        self.current_index = index

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read one BGR frame from the active camera."""

        if self._capture is None or not self._capture.isOpened():
            return False, None
        ok, frame = self._capture.read()
        return bool(ok), frame if ok else None

    @property
    def is_open(self) -> bool:
        """Whether a capture device is active."""

        return bool(self._capture is not None and self._capture.isOpened())

    def release(self) -> None:
        """Release the current device, if any."""

        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.current_index = None

    def __enter__(self) -> "CameraManager":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
