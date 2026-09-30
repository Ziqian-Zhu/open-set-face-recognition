"""Read AVFoundation metadata without creating a capture session or requesting access.

The video + muxed array and uniqueID comparison match OpenCV 4.14's
cap_avfoundation_mac.mm. No PyObjC package or Swift compiler is required.
"""

import ctypes as C
from functools import cmp_to_key


def list_video_devices():
    av = C.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
    objc = C.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [C.c_char_p]
    objc.objc_getClass.restype = C.c_void_p
    objc.sel_registerName.argtypes = [C.c_char_p]
    objc.sel_registerName.restype = C.c_void_p

    def send(receiver, selector, result=C.c_void_p, arguments=(), values=()):
        function = C.CFUNCTYPE(result, C.c_void_p, C.c_void_p, *arguments)(
            ("objc_msgSend", objc))
        return function(receiver, objc.sel_registerName(selector.encode()), *values)

    def string(receiver):
        raw = send(receiver, "UTF8String", C.c_char_p)
        return raw.decode("utf-8") if raw else ""

    pool = send(objc.objc_getClass(b"NSAutoreleasePool"), "new")
    try:
        capture_class = objc.objc_getClass(b"AVCaptureDevice")
        if not capture_class:
            raise RuntimeError("系统 AVFoundation 设备接口不可用")
        devices = []
        for media in ("AVMediaTypeVideo", "AVMediaTypeMuxed"):
            array = send(capture_class, "devicesWithMediaType:", arguments=(C.c_void_p,),
                         values=(C.c_void_p.in_dll(av, media).value,))
            for index in range(send(array, "count", C.c_ulong)):
                device = send(array, "objectAtIndex:", arguments=(C.c_ulong,), values=(index,))
                devices.append((device, send(device, "uniqueID")))
        devices.sort(key=cmp_to_key(lambda a, b: send(
            a[1], "compare:", C.c_long, (C.c_void_p,), (b[1],))))
        return [dict(index=index, name=string(send(device, "localizedName")),
                     unique_id=string(uid), device_type=string(send(device, "deviceType")),
                     connected=bool(send(device, "isConnected", C.c_bool)))
                for index, (device, uid) in enumerate(devices)]
    finally:
        send(pool, "drain", None)
