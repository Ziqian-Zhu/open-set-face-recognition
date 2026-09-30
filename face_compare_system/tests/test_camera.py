"""Camera routing tests never start real capture hardware."""

from unittest.mock import Mock

import pytest

from face_compare.camera import CameraManager
from face_compare.config import CameraConfig


def device(index, name, kind, uid, connected=True):
    return dict(index=index, name=name, device_type=kind, unique_id=uid, connected=connected)


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr("face_compare.camera.platform.system", lambda: "Darwin")
    enumeration = Mock(return_value=[
        device(0, "iPhone相机", "AVCaptureDeviceTypeExternalUnknown", "phone"),
        device(1, "FaceTime高清摄像头", "AVCaptureDeviceTypeBuiltInWideAngleCamera", "internal"),
        device(2, "USB Camera", "AVCaptureDeviceTypeExternal", "usb"),
        device(3, "Desk View", "AVCaptureDeviceTypeDeskViewCamera", "desk"),
        device(4, "Another phone", "AVCaptureDeviceTypeContinuityCamera", "phone2"),
        device(5, "Unplugged", "AVCaptureDeviceTypeExternal", "absent", False),
    ])
    monkeypatch.setattr("face_compare.macos_devices.list_video_devices", enumeration)
    capture = Mock()
    capture.isOpened.return_value = True
    constructor = Mock(return_value=capture)
    monkeypatch.setattr("face_compare.camera.cv2.VideoCapture", constructor)
    return enumeration, constructor, capture


def test_mac_discovery_does_not_open_any_camera(mac):
    enumeration, constructor, _ = mac
    devices = CameraManager(CameraConfig()).discover()
    assert [item.index for item in devices] == [1, 2]
    assert devices[0].built_in
    assert "FaceTime" in devices[0].label
    constructor.assert_not_called()


def test_mac_open_remaps_stable_identity_after_index_change(mac):
    enumeration, constructor, capture = mac
    manager = CameraManager(CameraConfig())
    selected = manager.discover()[0]
    enumeration.return_value = [device(0, "FaceTime", "BuiltInWideAngleCamera", "internal")]
    manager.open(selected.index, unique_id=selected.unique_id)
    assert constructor.call_args.args[0] == 0
    assert manager.current_index == 0
    capture.set.assert_not_called()


def test_disconnected_selection_never_falls_back_to_phone(mac):
    enumeration, constructor, _ = mac
    enumeration.return_value = [device(1, "iPhone", "ExternalUnknown", "phone")]
    with pytest.raises(RuntimeError, match="不会自动切换"):
        CameraManager(CameraConfig()).open(1, unique_id="internal")
    constructor.assert_not_called()


def test_explicit_phone_index_is_blocked(mac):
    with pytest.raises(RuntimeError):
        CameraManager(CameraConfig()).open(0)
    mac[1].assert_not_called()


def test_metadata_failure_never_triggers_hardware_probe(mac):
    mac[0].side_effect = RuntimeError("metadata unavailable")
    with pytest.raises(RuntimeError):
        CameraManager(CameraConfig()).discover()
    mac[1].assert_not_called()


def test_failed_open_releases_capture(mac):
    mac[2].isOpened.return_value = False
    manager = CameraManager(CameraConfig())
    with pytest.raises(RuntimeError, match="无法打开"):
        manager.open(1)
    mac[2].release.assert_called_once()
    assert not manager.is_open
