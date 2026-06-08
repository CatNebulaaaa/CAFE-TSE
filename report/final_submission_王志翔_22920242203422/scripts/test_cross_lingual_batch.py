"""Batch cross-lingual test with multiple speaker pairs (5 per direction).

Expands the single-pair test_cross_lingual.py to 5 pairs per direction,
reporting per-pair metrics plus mean/median/min/max aggregates.
"""

import sys, torch, argparse, time, json, csv, statistics, subprocess, re
from pathlib import Path
from collections import Counter

SR_TARGET = 8000
SEG_SEC = 4.0
SEG = int(SR_TARGET * SEG_SEC)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "speakerbeam" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "asteroid_site"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_open_speakerbeam import build_model
from cafe_tse.utils.audio_io import read_wav, fix_length
from cafe_tse.losses.sisdr import si_sdr


def rms_norm(wav, target_rms=0.05):
    return wav * (target_rms / wav.pow(2).mean().sqrt().clamp_min(1e-8))


def mix_at_snr(signal, noise, snr_db=0.0):
    sig_pow = signal.pow(2).mean().clamp_min(1e-10)
    noi_pow = noise.pow(2).mean().clamp_min(1e-10)
    scale = (sig_pow / (10 ** (snr_db / 10.0)) / noi_pow).sqrt()
    return signal + scale * noise


def run_pair(model, device, target_path, enroll_path, interferer_path, label):
    tgt, _ = read_wav(target_path, target_sr=SR_TARGET)
    enr, _ = read_wav(enroll_path, target_sr=SR_TARGET)
    itf, _ = read_wav(interferer_path, target_sr=SR_TARGET)
    tgt = fix_length(tgt, SEG)
    enr = fix_length(enr, SEG)
    itf = fix_length(itf, SEG)
    mixture = mix_at_snr(tgt, itf, snr_db=0.0)
    mix_norm = rms_norm(mixture).unsqueeze(0).to(device)
    enr_norm = rms_norm(enr).unsqueeze(0).to(device)
    tgt_norm = rms_norm(tgt)
    with torch.no_grad():
        est = model(mix_norm, enr_norm)
    if est.dim() == 3:
        est = est.squeeze(1)
    s = float(si_sdr(est.cpu(), tgt_norm.unsqueeze(0)).item())
    si = float(s - si_sdr(mix_norm.cpu(), tgt_norm.unsqueeze(0)).item())
    return {"label": label, "si_sdr": round(s, 4), "si_sdri": round(si, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thchs30_tgz", default="data/downloads/data_thchs30.tgz")
    parser.add_argument("--librispeech_root",
                        default="data/raw/LibriSpeech/LibriSpeech/test-clean")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(argparse.Namespace(**ckpt["args"])).to(device)
    model.eval()

    # -- Chinese speakers from THCHS-30 --
    print("=== Extracting Chinese speakers from THCHS-30 ===")
    # Use cached file list or create one
    cache_path = Path("/tmp/thchs30_filelist.txt")
    if cache_path.exists():
        tar_out = cache_path.read_text()
        print(f"Using cached file list ({len(tar_out.splitlines())} lines)")
    else:
        tar_out = subprocess.check_output(["tar", "tzf", args.thchs30_tgz]).decode()
        cache_path.write_text(tar_out)
    zh_speakers = set()
    zh_files = []
    for line in tar_out.split("\n"):
        if line.endswith(".wav"):
            base = line.split("/")[-1].replace(".wav", "")
            m = re.match(r"([A-C]\d+)", base)
            if m:
                zh_speakers.add(m.group(1))
                zh_files.append((m.group(1), line))

    zh_spk_counts = Counter(s for s, _ in zh_files)
    # Pick N pairs for target (speakers with most files)
    zh_sel = [s for s, _ in zh_spk_counts.most_common(args.n_pairs + 5)
              if s not in ["A23", "B33"]][:args.n_pairs]
    zh_pairs = {}
    for spk in zh_sel:
        files = [f for s, f in zh_files if s == spk]
        zh_pairs[spk] = [files[0], files[1]]
        for f in [files[0], files[1]]:
            subprocess.run(["tar", "xzf", args.thchs30_tgz, "-C", "/tmp", f],
                           capture_output=True)
    print(f"ZH target/enroll speakers: {zh_sel}")

    # Pick N different speakers for interferers
    zh_int_sel = [s for s, _ in zh_spk_counts.most_common(args.n_pairs + 10)
                  if s not in zh_sel and s not in ["A23", "B33"]][:args.n_pairs]
    zh_int_files = {}
    for spk in zh_int_sel:
        files = [f for s, f in zh_files if s == spk]
        zh_int_files[spk] = files[0]
        subprocess.run(["tar", "xzf", args.thchs30_tgz, "-C", "/tmp", files[0]],
                       capture_output=True)
    print(f"ZH interferer speakers: {zh_int_sel}")

    # -- English speakers from LibriSpeech --
    print("\n=== Finding English speakers from LibriSpeech ===")
    raw_ls = Path(args.librispeech_root)
    en_speakers = sorted([d for d in raw_ls.iterdir()
                          if d.is_dir() and d.name.isdigit()])
    en_sel = [s for s in en_speakers
              if s.name not in ["121", "1089"]][:args.n_pairs + 1]
    en_pairs = {}
    for spk in en_sel[:args.n_pairs]:
        files = sorted(spk.rglob("*.flac"))
        en_pairs[spk.name] = [str(files[0]), str(files[1])]
    print(f"EN target/enroll speakers: {list(en_pairs.keys())}")

    # English interferers from an extra speaker
    en_int_spk = en_sel[args.n_pairs]
    en_int_flac = sorted(en_int_spk.rglob("*.flac"))
    print(f"EN interferer speaker: {en_int_spk.name}")

    # -- Run Tests --
    rows = []

    print("\n=== Direction A: ZH target + EN interferer ===")
    for i in range(args.n_pairs):
        zh_spk = zh_sel[i]
        en_spk = list(en_pairs.keys())[i]
        zh_tgt = f"/tmp/data_thchs30/train/{zh_pairs[zh_spk][0].split('/')[-1]}"
        zh_enr = f"/tmp/data_thchs30/train/{zh_pairs[zh_spk][1].split('/')[-1]}"
        en_itf = en_int_flac[i]  # use different utterance per pair
        r = run_pair(model, device, zh_tgt, zh_enr, en_itf,
                     f"zh_{zh_spk}_en_{en_spk}")
        r["direction"] = "zh_target_en_interferer"
        r["zh_speaker"] = zh_spk
        r["en_speaker"] = en_spk
        rows.append(r)
        print(f"  {r['label']}: SI-SDR={r['si_sdr']:.2f}, SI-SDRi={r['si_sdri']:.2f}")

    print("\n=== Direction B: EN target + ZH interferer ===")
    for i in range(args.n_pairs):
        en_spk = list(en_pairs.keys())[i]
        zh_spk = zh_int_sel[i]
        en_tgt = en_pairs[en_spk][0]
        en_enr = en_pairs[en_spk][1]
        zh_itf = f"/tmp/data_thchs30/train/{zh_int_files[zh_spk].split('/')[-1]}"
        r = run_pair(model, device, en_tgt, en_enr, zh_itf,
                     f"en_{en_spk}_zh_{zh_spk}")
        r["direction"] = "en_target_zh_interferer"
        r["en_speaker"] = en_spk
        r["zh_speaker"] = zh_spk
        rows.append(r)
        print(f"  {r['label']}: SI-SDR={r['si_sdr']:.2f}, SI-SDRi={r['si_sdri']:.2f}")

    # -- Save --
    fieldnames = ["label", "direction", "zh_speaker", "en_speaker",
                  "si_sdr", "si_sdri"]
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    def agg(rows_list, name):
        vals_s = [r["si_sdr"] for r in rows_list]
        vals_i = [r["si_sdri"] for r in rows_list]
        return {
            "direction": name, "n": len(rows_list),
            "si_sdr_mean": round(statistics.mean(vals_s), 4),
            "si_sdr_median": round(statistics.median(vals_s), 4),
            "si_sdr_min": round(min(vals_s), 4),
            "si_sdr_max": round(max(vals_s), 4),
            "si_sdri_mean": round(statistics.mean(vals_i), 4),
            "si_sdri_median": round(statistics.median(vals_i), 4),
        }

    zh_en_rows = [r for r in rows if r["direction"] == "zh_target_en_interferer"]
    en_zh_rows = [r for r in rows if r["direction"] == "en_target_zh_interferer"]
    summary = {
        "zh_target_en_interferer": agg(zh_en_rows, "zh_target_en_interferer"),
        "en_target_zh_interferer": agg(en_zh_rows, "en_target_zh_interferer"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items()}
                      for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
