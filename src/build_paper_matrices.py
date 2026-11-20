"""
Same post-download preprocessing as Yelp(LU_STW).ipynb and Amazon(LU_STW).ipynb.
Does not change those notebooks or any saved dataset file.
"""

import os
import subprocess
import pandas as pd


def _gdown(file_id, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print("exists", dest)
        return dest
    if os.path.exists(dest):
        os.remove(dest)
    url = "https://drive.google.com/uc?id=" + file_id
    subprocess.check_call(["gdown", url, "-O", dest])
    return dest


def build_yelp(root="/content"):
    """Veterinarians -> Pet Services. Same filters as the Yelp notebook."""
    biz = _gdown("1HKw8G4y1u2H_YITMczkQnRLt_Ysreb63", os.path.join(root, "yelp_academic_dataset_business.json"))
    rev = _gdown("1r9OrcBZRutvjIU9kEW6r1TLDn7vbrFY0", os.path.join(root, "yelp_academic_dataset_review.json"))
    df_business = pd.read_json(biz, lines=True)
    df_review = pd.read_json(rev, lines=True)[["user_id", "business_id", "stars"]]
    df_review = df_review.merge(
        df_business[["business_id", "categories"]].rename(columns={"categories": "category"}),
        on="business_id", how="left",
    )
    df_review = df_review[["user_id", "business_id", "category", "stars"]]
    df_active = df_review.groupby("user_id").filter(lambda grp: len(grp) > 100)
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
    print("yelp", R_s.shape, int((R_s > 0).sum()), R_t.shape, int((R_t > 0).sum()))
    return R_s, R_t


def parse_amazon_txt_file(file_path):
    product_ids, titles, prices, user_ids = [], [], [], []
    profile_names, helpfulnesses, scores, times, summaries, texts = [], [], [], [], [], []
    cur = dict(pid=None, title=None, price=None, uid=None, name=None, help=None, score=None, time=None, summary=None, text=None)
    in_review = False

    def flush():
        if cur["uid"] is None:
            return
        product_ids.append(cur["pid"]); titles.append(cur["title"]); prices.append(cur["price"])
        user_ids.append(cur["uid"]); profile_names.append(cur["name"]); helpfulnesses.append(cur["help"])
        scores.append(cur["score"]); times.append(cur["time"]); summaries.append(cur["summary"]); texts.append(cur["text"])

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("product/productId:"):
                flush()
                cur.update(uid=None, name=None, help=None, score=None, time=None, summary=None, text=None)
                cur["pid"] = line.split(":", 1)[1].strip()
                in_review = False
            elif line.startswith("product/title:"):
                cur["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("product/price:"):
                cur["price"] = line.split(":", 1)[1].strip()
            elif line.startswith("review/userId:"):
                flush()
                cur["uid"] = line.split(":", 1)[1].strip()
                in_review = True
            elif in_review and line.startswith("review/profileName:"):
                cur["name"] = line.split(":", 1)[1].strip()
            elif in_review and line.startswith("review/helpfulness:"):
                cur["help"] = line.split(":", 1)[1].strip()
            elif in_review and line.startswith("review/score:"):
                cur["score"] = line.split(":", 1)[1].strip()
            elif in_review and line.startswith("review/time:"):
                cur["time"] = line.split(":", 1)[1].strip()
            elif in_review and line.startswith("review/summary:"):
                cur["summary"] = line.split(":", 1)[1].strip()
            elif in_review and line.startswith("review/text:"):
                cur["text"] = line.split(":", 1)[1].strip()
    flush()
    return pd.DataFrame({
        "productId": product_ids, "title": titles, "price": prices, "userId": user_ids,
        "profileName": profile_names, "helpfulness": helpfulnesses, "score": scores,
        "time": times, "summary": summaries, "text": texts,
    })


def build_amazon(root="/content"):
    """Video Games -> Toys & Games. Same user filter as the Amazon notebook."""
    toys = _gdown("1yki3wBoOBYXrQS8vlWZJGv1vWyBJRfFU", os.path.join(root, "Toys_&_Games.txt"))
    games = _gdown("1PjnAcRXHUq9EHyMX5abvTL2urVjL1G_m", os.path.join(root, "Video_Games.txt"))
    df_ty = parse_amazon_txt_file(toys)
    df_vg = parse_amazon_txt_file(games)
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
    print("amazon", R_s.shape, int((R_s > 0).sum()), R_t.shape, int((R_t > 0).sum()))
    return R_s, R_t
