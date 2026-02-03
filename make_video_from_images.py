# make_video_from_images.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

OUT_DIR = Path("outputs")
FINAL_DIR = OUT_DIR / "images" / "final"
WORK_DIR = OUT_DIR / "video_work"

VIDEO_W = int(os.getenv("VIDEO_W", "1080"))
VIDEO_H = int(os.getenv("VIDEO_H", "1920"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))

TOTAL_SECONDS = 60

# 8구간(총 60초)
SEGMENTS = [
    ("01", 7.5),
    ("02", 7.5),
    ("03", 7.5),
    ("04", 7.5),
    ("05", 7.5),
    ("06", 7.5),
    ("07", 7.5),
    ("08", 7.5),
]

BURN_ONSCREEN_TEXT = os.getenv("BURN_ONSCREEN_TEXT", "1") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    log("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def find_fontfile() -> str:
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


def load_onscreen_texts_8() -> list[str]:
    """
    shorts.json beats(6개) -> 8개로 변환
    1) topic 또는 title_short
    2~7) beats[0..5].onscreen
    8) 고정 CTA
    """
    shorts = load_shorts_json()
    topic = (shorts.get("topic") or shorts.get("title_short") or "").strip()
    topic = topic[:12] if topic else "오늘 연예 랭킹"

    beats = shorts.get("beats")
    beat_texts: list[str] = []
    if isinstance(beats, list):
        for b in beats[:6]:
            t = ((b or {}).get("onscreen") or "").strip()
            beat_texts.append((t[:12] if t else " "))

    while len(beat_texts) < 6:
        beat_texts.append(" ")

    cta = "구독/내일도"
    return [topic] + beat_texts + [cta]


def pick_images_8() -> list[Path]:
    imgs: list[Path] = []
    for i in range(1, 9):
        matches: list[Path] = []
        for ext in ("png", "jpg", "jpeg", "webp"):
            matches.extend(FINAL_DIR.glob(f"{i:02d}.{ext}"))
        if not matches:
            raise RuntimeError(f"필수 이미지가 없습니다: {FINAL_DIR}/{i:02d}.*")
        imgs.append(matches[0])
    return imgs


def clear_workdir() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for p in WORK_DIR.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass


def find_voice_file() -> Path | None:
    # 1) 고정 파일명 우선
    for p in (OUT_DIR / "voice.mp3", OUT_DIR / "voice.m4a", OUT_DIR / "voice.wav"):
        if p.exists():
            return p

    # 2) 혹시 예전 형식(voice_YYYY...mp3)이 남아있다면 최신 1개
    cand = sorted(OUT_DIR.glob("voice_*.mp3"), key=lambda x: x.stat().st_mtime, reverse=True)
    if cand:
        return cand[0]

    return None


def make_video_from_existing_images() -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg를 찾지 못했습니다. Actions에서 apt-get install ffmpeg 필요")

    if not FINAL_DIR.exists():
        raise RuntimeError(f"{FINAL_DIR} 폴더가 없습니다. 먼저 이미지 생성 런을 성공시켜야 합니다.")

    imgs = pick_images_8()
    texts = load_onscreen_texts_8()

    fontfile = find_fontfile()
    if BURN_ONSCREEN_TEXT and not fontfile:
        raise RuntimeError("한글 폰트 파일을 찾지 못했습니다. Actions에서 fonts-noto-cjk 설치 필요")

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
        run_cmd(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(silent_video)],
            cwd=WORK_DIR,
        )
    except Exception:
        run_cmd(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(VIDEO_FPS), str(silent_video)],
            cwd=WORK_DIR,
        )

    voice = find_voice_file()
    final_video = OUT_DIR / "video_from_images.mp4"

    if voice:
        # 오디오가 짧으면 60초까지 무음 패딩(apad), 길면 -t 60으로 자름
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-i", str(voice),
            "-filter_complex", f"[1:a]apad=pad_dur={TOTAL_SECONDS}[a]",
            "-map", "0:v:0",
            "-map", "[a]",
            "-t", str(TOTAL_SECONDS),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(final_video),
        ])
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
        "timeline": SEGMENTS,
        "total_seconds": TOTAL_SECONDS,
        "burn_onscreen_text": BURN_ONSCREEN_TEXT,
    }
    (OUT_DIR / "video_from_images_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return final_video


if __name__ == "__main__":
    out = make_video_from_existing_images()
    log(f"[OK] {out}")
