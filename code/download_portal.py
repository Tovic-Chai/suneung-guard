"""공공데이터포털 파일 데이터를 웹 화면과 같은 순서로 내려받는다.

1) 상세 페이지에서 다운로드 단추의 인자(fn_fileDataDown)를 읽고
2) /tcs/dss/selectFileDataDownload.do 로 파일 정보를 받고
3) /cmm/cmm/check-limit.json 으로 제한 여부를 확인한 뒤 (보안문자가 필요하면 멈춤)
4) /cmm/cmm/fileDownload.do 로 받는다.

실행:  py tools/download_portal.py 15098904 15098903 ...
저장:  data/raw/<번호>_<원래 파일 이름>
"""
from __future__ import annotations

import http.cookiejar
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
BASE = "https://www.data.go.kr"
UA = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def get(url: str) -> bytes:
    return opener.open(urllib.request.Request(url, headers=UA), timeout=60).read()


def post(url: str, data: dict) -> bytes:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers={**UA, "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"})
    return opener.open(req, timeout=60).read()


def download(pk: str) -> None:
    page = get(f"{BASE}/data/{pk}/fileData.do").decode("utf-8", "ignore")
    m = re.search(r"fn_fileDataDown\('(\d+)',\s*'([^']+)',\s*'([^']*)',\s*'(\d+)',\s*'(\d+)'\)", page)
    if not m:
        print(f"{pk}: 다운로드 단추를 찾지 못함")
        return
    public_pk, detail_pk, atch, sn, _hist = m.groups()
    info = json.loads(post(f"{BASE}/tcs/dss/selectFileDataDownload.do", {
        "publicDataDetailPk": detail_pk, "publicDataPk": public_pk, "atchFileId": atch,
        "fileDetailSn": sn, "publicDataTyCode": "PR0051"}).decode("utf-8", "ignore"))
    if not info.get("status"):
        print(f"{pk}: 파일 정보 실패 {info.get('error')}")
        return
    atch_id, file_sn = info["atchFileId"], info["fileDetailSn"]
    name = info["dataSetFileDetailInfo"].get("dataNm") or info["dataSetFileDetailInfo"].get("publicDataSj") or pk
    limit = json.loads(post(f"{BASE}/cmm/cmm/check-limit.json", {"atchFileId": atch_id, "fileDetailSn": file_sn}).decode())
    if limit.get("needCaptcha"):
        print(f"{pk}: 보안문자가 필요해 멈춤. 브라우저에서 직접 받아야 함 ({BASE}/data/{pk}/fileData.do)")
        return
    url = f"{BASE}/cmm/cmm/fileDownload.do?atchFileId={atch_id}&fileDetailSn={file_sn}&dataNm={urllib.parse.quote(name)}"
    with opener.open(urllib.request.Request(url, headers=UA), timeout=300) as r:
        cd = r.headers.get("Content-Disposition", "")
        fn = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd)
        fname = urllib.parse.unquote(fn.group(1)) if fn else f"{name}.bin"
        try:  # 헤더가 latin-1로 읽힌 UTF-8 이름이면 바로잡는다
            fname = fname.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        data = r.read()
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{pk}_{fname}"
    out.write_bytes(data)
    print(f"{pk}: {out.name} ({len(data):,} bytes)")


if __name__ == "__main__":
    for pk in sys.argv[1:]:
        try:
            download(pk)
        except Exception as e:
            print(f"{pk}: 오류 {type(e).__name__}: {e}")
        time.sleep(1.0)
