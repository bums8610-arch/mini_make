from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

OUT_DIR = Path("outputs")
FINAL_DIR = OUT_DIR / "images" / "final"
WORK_DIR = OUT_DIR / "video_work2"
WORK_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_W = int(os.getenv("VIDEO_W", "1080"))
VIDEO_H = int(os.getenv("VIDEO_H", "1920"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
BURN_ONSCREEN_TEXT = os.getenv("BURN_ONSCREEN_TEXT", "1").strip() == "1"

# 60초 타임라인(6구간) = 6장 이미지 사용
SEGMENTS = [
    ("0-2s", 2),
    ("2-10s", 8),
    ("10-25s", 15),
    ("25-40s", 15),
    ("40-55s", 15),
    ("55-60s", 5),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    log("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def find_fontfile() -> str:
    # env로 지정 가능
    env_font = os.getenv("FONTFILE", "").strip()
    if env_font and Path(env_font).exists():
        return env_font

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p

    # fallback: 아무 폰트나
    for p in Path("/usr/share/fonts").rglob("*"):
        if p.suffix.lower() in (".ttf", ".ttc", ".otf"):
            return str(p)
    return ""


def load_shorts_json() -> dict[str, Any]:
    p = OUT_DIR / "shorts.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_onscreen_texts() -> list[str]:
    shorts = load_shorts_json()
    beats = shorts.get("beats")
    if isinstance(beats, list) and len(beats) >= 6:
        out = []
        for b in beats[:6]:
            t = (b or {}).get("onscreen") or ""
            out.append(str(t).strip()[:12] or " ")
        return out
    # 없으면 기본값
    return [" ", " ", " ", " ", " ", " "]


def pick_segment_images() -> list[Path]:
    # 01.* ~ 06.* 찾기
    imgs: list[Path] = []
    for i in range(1, 7):
        matches = []
        for ext in ("png", "jpg", "jpeg", "webp"):
            matches.extend(FINAL_DIR.glob(f"{i:02d}.{ext}"))
        if not matches:
            raise RuntimeError(f"필수 이미지가 없습니다: {FINAL_DIR}/{i:02d}.*")
        imgs.append(matches[0])
    return imgs


def clear_workdir() -> None:
    for p in WORK_DIR.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass


def make_video() -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg를 찾지 못했습니다. Actions에서 apt-get install ffmpeg 필요")

    if not FINAL_DIR.exists():
        raise RuntimeError(f"{FINAL_DIR} 폴더가 없습니다. 먼저 이미지 생성 런을 성공시켜야 합니다.")

    imgs = pick_segment_images()
    texts = load_onscreen_texts()

    fontfile = find_fontfile()
    if BURN_ONSCREEN_TEXT and not fontfile:
        raise RuntimeError("폰트 파일을 찾지 못했습니다. Actions에서 fonts-noto-cjk 설치 필요")

    clear_workdir()

    seg_files: list[Path] = []
    for idx, ((slot, dur), img, ons) in enumerate(zip(SEGMENTS, imgs, texts), start=1):
        textfile = WORK_DIR / f"text_{idx:02d}.txt"
        textfile.write_text(ons if ons else " ", encoding="utf-8")

        seg_out = WORK_DIR / f"seg_{idx:02d}.mp4"

        vf = (
            f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_W}:{VIDEO_H},format=yuv420p"
        )
        if BURN_ONSCREEN_TEXT:
            vf += (
                f",drawtext=fontfile='{fontfile}':textfile='{textfile}':"
                f"x=(w-text_w)/2:y=h*0.12:fontsize=64:fontcolor=white:"
                f"box=1:boxcolor=black@0.35:boxborderw=22"
            )

        run_cmd([
            "ffmpeg", "-y",
            "-loop", "1",
            "-t", str(dur),
            "-i", str(img),
            "-vf", vf,
            "-r", str(VIDEO_FPS),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(seg_out),
        ])
        seg_files.append(seg_out)

    concat_list = WORK_DIR / "concat.txt"
    concat_list.write_text("\n".join([f"file '{p.name}'" for p in seg_files]) + "\n", encoding="utf-8")

    silent_video = OUT_DIR / "video_silent_from_images.mp4"
    try:
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(silent_video)], cwd=WORK_DIR)
    except Exception:
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(VIDEO_FPS), str(silent_video)], cwd=WORK_DIR)

    # 음성 파일이 있으면 합치기, 없으면 무음 영상 그대로
    voice = None
    for candidate in (OUT_DIR / "voice.mp3", OUT_DIR / "voice.m4a", OUT_DIR / "voice.wav"):
        if candidate.exists():
            voice = candidate
            break

    final_video = OUT_DIR / "video_from_images.mp4"
    if voice:
        try:
            run_cmd(["ffmpeg", "-y", "-i", str(silent_video), "-i", str(voice),
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final_video)])
        except Exception:
            run_cmd(["ffmpeg", "-y", "-i", str(silent_video), "-i", str(voice),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final_video)])
    else:
        final_video.write_bytes(silent_video.read_bytes())

    manifest = {
        "ok": True,
        "final_video": str(final_video),
        "silent_video": str(silent_video),
        "used_images": [str(p) for p in imgs],
        "used_texts": texts,
        "voice_used": str(voice) if voice else "",
        "fps": VIDEO_FPS,
        "size": [VIDEO_W, VIDEO_H],
        "burn_onscreen_text": BURN_ONSCREEN_TEXT,
    }
    (OUT_DIR / "video_from_images_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return final_video


if __name__ == "__main__":
    out = make_video()
    log(f"[OK] {out}")

