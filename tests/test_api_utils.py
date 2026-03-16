"""Tests for dart_mlci.api.utils edge cases."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from dart_mlci.api.utils import array_to_base64_png, base64_to_array


class TestBase64ToArrayEdgeCases:
    """Edge cases for base64_to_array."""

    def test_data_uri_missing_comma(self):
        """Data URI without comma should raise ValueError."""
        with pytest.raises(ValueError, match="missing comma"):
            base64_to_array("data:image/png;base64")

    def test_channel_first_format(self):
        """CxHxW input should be transposed to HxWx3."""
        # Create a CxHxW image (3, 32, 32)
        arr = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        # Transpose to HxWxC for PIL
        arr_hwc = np.transpose(arr, (1, 2, 0))
        img = Image.fromarray(arr_hwc, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = base64_to_array(b64)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_single_channel_hwx1(self):
        """HxWx1 image should be expanded to HxWx3."""
        gray = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        img = Image.fromarray(gray, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = base64_to_array(b64)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_rgba_to_rgb(self):
        """RGBA (HxWx4) should be converted to HxWx3."""
        rgba = np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = base64_to_array(b64)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_float_image_normalization(self):
        """Float image with values in [0,1] should be normalized to uint8."""
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        # Save as uint8 PNG (can't save float directly)
        arr_uint8 = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = base64_to_array(b64)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_empty_decoded_image(self):
        """Empty base64 payload should raise ValueError."""
        # Base64 encode empty bytes
        b64 = base64.b64encode(b"").decode()
        with pytest.raises(ValueError):
            base64_to_array(b64)


class TestArrayToBase64PngEdgeCases:
    """Edge cases for array_to_base64_png."""

    def test_channel_first_conversion(self):
        """CxHxW array should be transposed before encoding."""
        arr = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        b64 = array_to_base64_png(arr)
        assert len(b64) > 0

        # Decode and verify shape
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        decoded = np.array(img)
        assert decoded.shape == (32, 32, 3)

    def test_single_channel_squeeze(self):
        """HxWx1 array should be squeezed to grayscale."""
        arr = np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8)
        b64 = array_to_base64_png(arr)
        assert len(b64) > 0

    def test_float_normalization(self):
        """Float array should be normalized to uint8."""
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        b64 = array_to_base64_png(arr)
        assert len(b64) > 0
