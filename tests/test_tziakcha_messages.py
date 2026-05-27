import base64
import json
import zlib

from advisor_service.messages import decode_observed_payload


def test_decode_plain_json_string():
    payload = {"kind": "text", "data": json.dumps({"m": 2, "r": 3})}
    assert decode_observed_payload(payload) == {"m": 2, "r": 3}


def test_decode_base64_zlib_json():
    raw = zlib.compress(json.dumps({"m": 2, "r": 7, "v": 12}).encode("utf-8"))
    payload = {"kind": "binary", "base64": base64.b64encode(raw).decode("ascii")}
    assert decode_observed_payload(payload) == {"m": 2, "r": 7, "v": 12}


def test_decode_base64_raw_deflate_json():
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = compressor.compress(json.dumps({"m": 2, "r": 6}).encode("utf-8"))
    raw += compressor.flush()
    payload = {"kind": "binary", "base64": base64.b64encode(raw).decode("ascii")}
    assert decode_observed_payload(payload) == {"m": 2, "r": 6}
