from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from src.bundle_loader import BundleError, load_bundle, read_npz


def test_excessive_entry_count():
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        for index in range(73):
            archive.writestr(f"entry_{index}", b"")
    with pytest.raises(BundleError, match="entry count"):
        load_bundle(payload.getvalue())


def test_encrypted_member_rejected():
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", b"{}")
    # Set the encryption flag in the central directory without encrypting data.
    data = bytearray(payload.getvalue())
    central_directory = data.index(b"PK\x01\x02")
    data[central_directory + 8] |= 1
    with pytest.raises(BundleError, match="suspicious ZIP member"):
        load_bundle(bytes(data))


def test_missing_required_array():
    payload = BytesIO()
    np.savez(payload, betas=np.zeros(10))
    with pytest.raises(BundleError, match="required arrays"):
        read_npz(payload.getvalue())
