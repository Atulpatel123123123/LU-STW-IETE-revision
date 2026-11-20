"""
Rebuild the paper's Yelp and Amazon matrices with the same preprocessing as
Yelp(LU_STW).ipynb and Amazon(LU_STW).ipynb, then check them against the
dataset table in the paper. A mismatch stops the run instead of ranking on a
different split.

Amazon: downloaded from the official SNAP source (same raw format as the
notebook's Video_Games.txt / Toys_&_Games.txt).
Yelp: the two Yelp JSON files must already exist in /content or in Google
Drive, because the Yelp dataset cannot be downloaded without accepting terms.
"""

import glob
import gzip
import json
import os
import urllib.request

import pandas as pd

SNAP = "https://snap.stanford.edu/data/amazon/"
AMAZON_SRC = "Video_Games.txt.gz"
AMAZON_TGT = "Toys_%26_Games.txt.gz"

# (users, source items, source ratings, target items, target ratings) from the paper
EXPECTED = {
    "yelp": (1225, 822, 2150, 1327, 3272),
    "amazon": (400, 12691, 24308, 12562, 15760),
}

SEARCH_ROOTS = ["/content", "/content/drive/MyDrive"]


def _check(pair, R_s, R_t):
    got = (R_s.shape[0], R_s.shape[1], int((R_s > 0).sum()), R_t.shape[1], int((R_t > 0).sum()))
    print(pair, "users/src items/src ratings/tgt items/tgt ratings:", got)
    if R_s.shape[0] != R_t.shape[0]:
        raise ValueError(f"{pair}: source and target user counts differ: {R_s.shape[0]} vs {R_t.shape[0]}")
    if got != EXPECTED[pair]:
        raise ValueError(
            f"{pair}: matrices do not match the paper. got {got}, paper {EXPECTED[pair]}. "
            "Ranking was not run."
        )
    print(pair, "matches the paper's dataset table")


def _find(names, roots=None):
    for root in roots or SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for name in names:
            direct = os.path.join(root, name)
            if os.path.isfile(direct):
                return direct
        for name in names:
            hits = glob.glob(os.path.join(root, "**", name), recursive=True)
            if hits:
                return hits[0]
    return None


def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("exists", dest)
        return dest
    print("downloading", url)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


# ----------------------------------------------------------------- Yelp
def _read_yelp_reviews(path):
    users, items, stars = [], [], []
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            users.append(d["user_id"])
            items.append(d["business_id"])
            stars.append(d["stars"])
    return pd.DataFrame({"user_id": users, "business_id": items, "stars": stars})


def build_yelp(business_path=None, review_path=None):
    """Veterinarians -> Pet Services, users with more than 100 reviews."""
    business_path = business_path or _find(["yelp_academic_dataset_business.json"])
    review_path = review_path or _find(["yelp_academic_dataset_review.json"])
    if not business_path or not review_path:
        raise FileNotFoundError(
            "Yelp files not found. Put yelp_academic_dataset_business.json and "
            "yelp_academic_dataset_review.json in /content or in Google Drive "
            "(run drive.mount first), or pass business_path= and review_path=."
        )
    print("yelp files:", business_path, review_path)
    df_business = pd.read_json(business_path, lines=True)[["business_id", "categories"]]
    df_review = _read_yelp_reviews(review_path)
    print("reviews", df_review.shape)
    df_review = df_review.merge(
        df_business.rename(columns={"categories": "category"}), on="business_id", how="left"
    )[["user_id", "business_id", "category", "stars"]]
    counts = df_review["user_id"].value_counts()
    df_active = df_review[df_review["user_id"].isin(counts[counts > 100].index)]
    mask_pet = df_active["category"].str.contains(r"\bPet Services\b", na=False)
    mask_vet = df_active["category"].str.contains(r"\bVeterinarians\b", na=False)
    common = set(df_active.loc[mask_pet, "user_id"]) & set(df_active.loc[mask_vet, "user_id"])
    df_pet = df_active[mask_pet & df_active["user_id"].isin(common)]
    df_vet = df_active[mask_vet & df_active["user_id"].isin(common)]
    matrix_pet = df_pet.pivot_table(index="user_id", columns="business_id", values="stars", fill_value=0)
    matrix_vet = df_vet.pivot_table(index="user_id", columns="business_id", values="stars", fill_value=0)
    users = matrix_vet.index.intersection(matrix_pet.index)
    R_s = matrix_vet.loc[users].fillna(0).to_numpy(dtype=float)
    R_t = matrix_pet.loc[users].fillna(0).to_numpy(dtype=float)
    _check("yelp", R_s, R_t)
    return R_s, R_t


# --------------------------------------------------------------- Amazon
def parse_amazon_txt_file(file_path):
    """Same record logic as parse_amazon_txt_file in Amazon(LU_STW).ipynb."""
    product_ids, user_ids, scores = [], [], []
    pid = uid = score = None
    in_review = False
    with _open_text(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("product/productId:"):
                if uid is not None:
                    product_ids.append(pid); user_ids.append(uid); scores.append(score)
                    uid = None
                    score = None
                pid = line.split(":", 1)[1].strip()
                in_review = False
            elif line.startswith("review/userId:"):
                if uid is not None:
                    product_ids.append(pid); user_ids.append(uid); scores.append(score)
                uid = line.split(":", 1)[1].strip()
                in_review = True
            elif in_review and line.startswith("review/score:"):
                score = line.split(":", 1)[1].strip()
    if uid is not None:
        product_ids.append(pid); user_ids.append(uid); scores.append(score)
    return pd.DataFrame({"productId": product_ids, "userId": user_ids, "score": scores})


def build_amazon(root="/content", games_path=None, toys_path=None):
    """Video Games -> Toys & Games, users with more than 20 ratings across both."""
    games_path = games_path or _find(["Video_Games.txt", "Video_Games.txt.gz"])
    toys_path = toys_path or _find(["Toys_&_Games.txt", "Toys_&_Games.txt.gz"])
    if not games_path:
        games_path = _download(SNAP + AMAZON_SRC, os.path.join(root, "Video_Games.txt.gz"))
    if not toys_path:
        toys_path = _download(SNAP + AMAZON_TGT, os.path.join(root, "Toys_&_Games.txt.gz"))
    print("amazon files:", games_path, toys_path)
    df_vg = parse_amazon_txt_file(games_path)
    df_ty = parse_amazon_txt_file(toys_path)
    print("rows", len(df_vg), len(df_ty))
    df_vg["score"] = pd.to_numeric(df_vg["score"], errors="coerce")
    df_ty["score"] = pd.to_numeric(df_ty["score"], errors="coerce")
    counts = pd.concat([df_vg[["userId"]], df_ty[["userId"]]])["userId"].value_counts()
    heavy = set(counts[counts > 20].index)
    common = sorted(heavy & set(df_vg["userId"]) & set(df_ty["userId"]))
    vg = df_vg[df_vg["userId"].isin(common)]
    ty = df_ty[df_ty["userId"].isin(common)]
    matrix_vg = vg.pivot_table(index="userId", columns="productId", values="score", fill_value=0)
    matrix_ty = ty.pivot_table(index="userId", columns="productId", values="score", fill_value=0)
    users = matrix_vg.index.intersection(matrix_ty.index)
    R_s = matrix_vg.loc[users].fillna(0).to_numpy(dtype=float)
    R_t = matrix_ty.loc[users].fillna(0).to_numpy(dtype=float)
    _check("amazon", R_s, R_t)
    return R_s, R_t
