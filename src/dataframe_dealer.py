import pandas as pd


data_path = "OCT_to_VF/data_table.xlsx"
dataset_ratio = [0.8, 0.1, 0.1]  # [train, validation, test]


def get_df_length(df):
    """Return the number of rows in the input DataFrame."""
    return len(df)


def analyze_all_md_ratios(df):
    """
    Analyze MD severity distribution over all valid samples.

    Severity bins:
    - Mild: MD > -6
    - Moderate: -12 <= MD <= -6
    - Severe: MD < -12
    """
    try:
        if 'HVF_MD' not in df.columns:
            return {"error": "Column 'HVF_MD' is missing."}

        md_df = df.copy()
        md_df['HVF_MD'] = pd.to_numeric(md_df['HVF_MD'], errors='coerce')
        valid_df = md_df.dropna(subset=['HVF_MD'])
        total_count = len(valid_df)

        if total_count == 0:
            return {"info": "No valid numeric HVF_MD values were found."}

        mild_count = int((valid_df['HVF_MD'] > -6).sum())
        severe_count = int((valid_df['HVF_MD'] < -12).sum())
        moderate_count = total_count - mild_count - severe_count

        return {
            "total_valid_samples": total_count,
            "distribution": {
                "mild": {
                    "count": mild_count,
                    "ratio": round(mild_count / total_count, 4),
                    "label": "Mild (> -6)",
                },
                "moderate": {
                    "count": moderate_count,
                    "ratio": round(moderate_count / total_count, 4),
                    "label": "Moderate (-12 to -6)",
                },
                "severe": {
                    "count": severe_count,
                    "ratio": round(severe_count / total_count, 4),
                    "label": "Severe (< -12)",
                },
            },
        }

    except FileNotFoundError:
        return {"error": "Input file was not found."}
    except Exception as e:
        return {"error": f"Unexpected error during MD analysis: {str(e)}"}


def analyze_and_get_labeld_data(df):
    """
    Split samples by PM-label availability and report MD distribution for labeled samples.

    Returns:
        filtered_df: samples with non-empty Meta-PM labels.
        empty_label_df: samples with empty Meta-PM labels.
    """
    try:
        required_columns = ['HVF_FileName', 'Meta-PM', 'HVF_MD']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return [], {"error": f"Missing required columns: {missing}"}

        filtered_df = df[df['Meta-PM'].notna()].copy()
        empty_label_df = df[df['Meta-PM'].isna()].copy()

        if filtered_df.empty:
            return [], {"info": "No samples with non-empty Meta-PM labels were found."}

        filtered_df['HVF_MD'] = pd.to_numeric(filtered_df['HVF_MD'], errors='coerce')
        valid_md_df = filtered_df.dropna(subset=['HVF_MD'])
        total_count = len(valid_md_df)

        mild_count = 0
        moderate_count = 0
        severe_count = 0

        for md in valid_md_df['HVF_MD']:
            if md > -6:
                mild_count += 1
            elif md >= -12:
                moderate_count += 1
            else:
                severe_count += 1

        all_data_result = analyze_all_md_ratios(df)

        if "error" in all_data_result:
            print(f"[Error] {all_data_result['error']}")
        elif "info" in all_data_result:
            print(f"[Info] {all_data_result['info']}")
        else:
            print(f"[OK] Global MD distribution analyzed (valid samples: {all_data_result['total_valid_samples']}).")
            dist = all_data_result['distribution']
            print(f"   - Mild: {dist['mild']['ratio']:.2%} ({dist['mild']['count']} eyes)")
            print(f"   - Moderate: {dist['moderate']['ratio']:.2%} ({dist['moderate']['count']} eyes)")
            print(f"   - Severe: {dist['severe']['ratio']:.2%} ({dist['severe']['count']} eyes)")

        if total_count > 0:
            stats = {
                "labeled_sample_count": total_count,
                "mild_ratio": round(mild_count / total_count, 4),
                "moderate_ratio": round(moderate_count / total_count, 4),
                "severe_ratio": round(severe_count / total_count, 4),
                "counts": {
                    "mild": mild_count,
                    "moderate": moderate_count,
                    "severe": severe_count,
                },
            }
            print("[OK] Labeled-subset statistics:\n", stats)

        return filtered_df, empty_label_df

    except FileNotFoundError:
        return [], {"error": "Input file was not found."}
    except Exception as e:
        return [], {"error": f"Unexpected error: {str(e)}"}


def stratified_split_by_md(df, ratios=[0.8, 0.1, 0.1], seed=42):
    """
    Stratify by MD severity and split into train/validation/test.

    Args:
        df: Input DataFrame (typically samples with empty Meta-PM labels).
        ratios: [train_ratio, val_ratio, test_ratio].
        seed: Random seed for reproducible shuffling.

    Returns:
        (train_df, val_df, test_df)
    """
    data = df.copy()
    data['HVF_MD'] = pd.to_numeric(data['HVF_MD'], errors='coerce')
    data = data.dropna(subset=['HVF_MD'])

    def get_severity(md):
        if md > -6:
            return 'Mild'
        if md >= -12:
            return 'Moderate'
        return 'Severe'

    data['Severity_Label'] = data['HVF_MD'].apply(get_severity)

    train_fragments = []
    val_fragments = []
    test_fragments = []

    for _, group in data.groupby('Severity_Label'):
        shuffled_group = group.sample(frac=1, random_state=seed)

        n_total = len(shuffled_group)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])

        train_subset = shuffled_group.iloc[:n_train]
        val_subset = shuffled_group.iloc[n_train:n_train + n_val]
        test_subset = shuffled_group.iloc[n_train + n_val:]

        train_fragments.append(train_subset)
        val_fragments.append(val_subset)
        test_fragments.append(test_subset)

    train_df = pd.concat(train_fragments).sample(frac=1, random_state=seed)
    val_df = pd.concat(val_fragments).sample(frac=1, random_state=seed)
    test_df = pd.concat(test_fragments).sample(frac=1, random_state=seed)

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = pd.read_excel(data_path)
    all_length = get_df_length(df)
    train_len, val_len, test_len = [int(r * all_length) for r in dataset_ratio]

    print("Starting dataset partition analysis...")
    with_label_df, no_label_df = analyze_and_get_labeld_data(df)

    train_with_label_len = len(with_label_df)
    train_no_label_len = train_len - train_with_label_len
    sub_dataset_ratio = [
        train_no_label_len / len(no_label_df),
        val_len / len(no_label_df),
        test_len / len(no_label_df),
    ]
    train_no_label_df, val_no_label_df, test_no_label_df = stratified_split_by_md(
        no_label_df,
        ratios=sub_dataset_ratio,
        seed=42,
    )

    train_df = pd.concat([with_label_df, train_no_label_df]).sample(frac=1, random_state=42)
    val_df = val_no_label_df
    test_df = test_no_label_df

    print("[OK] Dataset split summary:")
    print(f"   [Train] Count: {len(train_df)} (target ratio: {dataset_ratio[0]:.2%})")
    print(f"       composition: {len(with_label_df)} (labeled) + {len(train_no_label_df)} (unlabeled)")
    print(f"   [Val]   Count: {len(val_df)} (target ratio: {dataset_ratio[1]:.2%})")
    print("       composition: unlabeled only")
    print(f"   [Test]  Count: {len(test_df)} (target ratio: {dataset_ratio[2]:.2%})")
    print("       composition: unlabeled only")
