import pandas as pd
import numpy as np
import scipy.io
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
import os
import warnings

# התעלמות מאזהרות
warnings.filterwarnings("ignore", category=UserWarning)


# --- פונקציות עזר ---

def robust_load_csv_v2(csv_path, debug=False):
    """
    טוען קובץ CSV עם מספר עמודות משתנה (כותרת צרה ומידע רחב).
    """
    if not os.path.exists(csv_path):
        if debug: print(f"[DEBUG] CSV file not found: {csv_path}")
        return None

    try:
        # זיהוי השורה שבה מתחיל המידע הרחב
        with open(csv_path, 'r', errors='ignore') as f:
            lines = f.readlines()

        if not lines:
            return None

        max_cols = 0
        data_start_line = 0

        # סריקת 30 השורות הראשונות
        for i, line in enumerate(lines[:30]):
            num_cols = len(line.split(','))
            if num_cols > max_cols:
                max_cols = num_cols
                data_start_line = i

        if max_cols < 10:
            if debug: print("[DEBUG] Max columns < 10. Not a thermal image matrix.")
            return None

        # טעינת הנתונים
        df = pd.read_csv(
            csv_path,
            skiprows=data_start_line,
            header=None,
            names=range(max_cols),
            engine='python'
        )

        # המרה למספרים וניקוי
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

        return df.values

    except Exception as e:
        if debug: print(f"[DEBUG] Error reading CSV: {e}")
        return None


def load_masks_debug(mat_path, shape, debug=False):
    """
    טוען מסכות. מטפל במקרה של משתנה 'labels' המכיל מטריצת לייבלים (1,2,3).
    """
    if not os.path.exists(mat_path):
        if debug: print(f"[DEBUG] Mask file not found: {mat_path}")
        return {}

    try:
        mat = scipy.io.loadmat(mat_path)
        masks = {}

        # בדיקה אם יש מפתחות מפורשים (Head/Body/Tail)
        for key in mat:
            if key.startswith('__'): continue
            key_lower = key.lower()
            if 'head' in key_lower:
                masks['Head'] = mat[key]
            elif 'body' in key_lower:
                masks['Body'] = mat[key]
            elif 'tail' in key_lower:
                masks['Tail'] = mat[key]

        # אם לא מצאנו מפתחות מפורשים, נבדוק אם יש משתנה 'labels'
        if not masks and 'labels' in mat:
            if debug: print("[DEBUG] Found 'labels' variable. Attempting to split by integer values (1,2,3).")
            lbl = mat['labels']

            # --- כאן מתבצע המיפוי - שנה את המספרים אם הסדר לא נכון ---
            # הנחה: 1=ראש, 2=גוף, 3=זנב
            masks['Head'] = (lbl == 1)
            masks['Body'] = (lbl == 2)
            masks['Tail'] = (lbl == 3)

        return masks

    except Exception as e:
        if debug: print(f"[DEBUG] Error loading MAT: {e}")
        return {}


def calculate_roi_features(temp_matrix, mask, roi_name):
    # לוגיקה להתאמת גדלים (חיתוך אם המסכה גדולה מהתמונה או להפך)
    if mask.shape != temp_matrix.shape:
        h_min = min(mask.shape[0], temp_matrix.shape[0])
        w_min = min(mask.shape[1], temp_matrix.shape[1])
        mask = mask[:h_min, :w_min]
        temp_matrix = temp_matrix[:h_min, :w_min]

    mask_bool = mask > 0  # המרה לבוליאני
    roi_pixels = temp_matrix[mask_bool]

    # החזרת NaN אם אין פיקסלים
    if len(roi_pixels) == 0:
        return {f'{roi_name}_{feat}': np.nan for feat in
                ['Mean', 'Var', 'Skew', 'Kurt', 'Entropy', 'Contrast', 'Corr', 'Homogeneity']}

    # 1. חישובים סטטיסטיים
    res = {}
    res[f'{roi_name}_Mean'] = np.mean(roi_pixels)
    res[f'{roi_name}_Var'] = np.var(roi_pixels)
    res[f'{roi_name}_Skew'] = skew(roi_pixels)
    res[f'{roi_name}_Kurt'] = kurtosis(roi_pixels, fisher=False)  # Pearson definition

    # נרמול ל-0-255 עבור GLCM ואנטרופיה
    min_t, max_t = temp_matrix.min(), temp_matrix.max()
    if max_t > min_t:
        norm_img = ((temp_matrix - min_t) / (max_t - min_t) * 255).astype(np.uint8)
    else:
        norm_img = np.zeros_like(temp_matrix, dtype=np.uint8)

    # 2. אנטרופיה
    roi_q = norm_img[mask_bool]
    if len(roi_q) > 0:
        counts, _ = np.histogram(roi_q, bins=256, range=(0, 256))
        p_i = counts / counts.sum()
        p_i = p_i[p_i > 0]
        res[f'{roi_name}_Entropy'] = -np.sum(p_i * np.log2(p_i))
    else:
        res[f'{roi_name}_Entropy'] = np.nan

    # 3. GLCM (Texture)
    # חיתוך Bounding Box לחיסכון בזמן עיבוד
    rows, cols = np.where(mask_bool)
    if len(rows) > 0:
        r1, r2 = rows.min(), rows.max() + 1
        c1, c2 = cols.min(), cols.max() + 1
        crop_img = norm_img[r1:r2, c1:c2]
        crop_mask = mask_bool[r1:r2, c1:c2]
        crop_img[~crop_mask] = 0

        try:
            g = graycomatrix(crop_img, [1], [0], levels=256, symmetric=True, normed=True)
            res[f'{roi_name}_Contrast'] = graycoprops(g, 'contrast')[0, 0]
            res[f'{roi_name}_Corr'] = graycoprops(g, 'correlation')[0, 0]
            res[f'{roi_name}_Homogeneity'] = graycoprops(g, 'homogeneity')[0, 0]
        except:
            res[f'{roi_name}_Contrast'] = np.nan
            res[f'{roi_name}_Corr'] = np.nan
            res[f'{roi_name}_Homogeneity'] = np.nan
    else:
        res[f'{roi_name}_Contrast'] = np.nan
        res[f'{roi_name}_Corr'] = np.nan
        res[f'{roi_name}_Homogeneity'] = np.nan

    return res


# --- MAIN ---

metadata_path = r"C:\Users\simto\OneDrive - Afeka College Of Engineering\פרויקט גמר\clean_metadata_all_experiments.xlsx"
output_path = r"C:\Users\simto\OneDrive - Afeka College Of Engineering\פרויקט גמר\Statistical_Features_Extracted_Final.xlsx"

print("--- Starting Extraction ---")
if not os.path.exists(metadata_path):
    print("CRITICAL ERROR: Metadata Excel file not found!")
    exit()

df = pd.read_excel(metadata_path)
print(f"Loaded metadata with {len(df)} rows.")

features_list = []
success_count = 0
debug_printed = False

for index, row in df.iterrows():
    csv_path = str(row.get('csv_Path')).strip().strip('"')
    mask_path = str(row.get('Mask_mat_Path')).strip().strip('"')

    if pd.isna(csv_path) or pd.isna(mask_path) or csv_path == 'nan':
        continue

    # Debug print for first file
    do_debug = False
    if not debug_printed:
        print(f"\n--- Debugging Row {index} ---")
        print(f"CSV: {os.path.basename(csv_path)}")
        print(f"Mask: {os.path.basename(mask_path)}")
        do_debug = True
        debug_printed = True

    # 1. טעינת תמונה
    temp_matrix = robust_load_csv_v2(csv_path, debug=do_debug)
    if temp_matrix is None:
        continue

    # 2. טעינת מסכות
    masks = load_masks_debug(mask_path, temp_matrix.shape, debug=do_debug)
    if not masks:
        if do_debug: print("[DEBUG] No masks extracted from file.")
        continue

    # 3. חילוץ מאפיינים
    row_data = row.to_dict()
    roi_found = False
    for roi in ['Head', 'Body', 'Tail']:
        if roi in masks:
            roi_found = True
            feats = calculate_roi_features(temp_matrix, masks[roi], roi)
            row_data.update(feats)
        else:
            for f in ['Mean', 'Var', 'Skew', 'Kurt', 'Entropy', 'Contrast', 'Corr', 'Homogeneity']:
                row_data[f'{roi}_{f}'] = np.nan

    if roi_found:
        features_list.append(row_data)
        success_count += 1
        if do_debug: print("[DEBUG] Success!")

    if index > 0 and index % 50 == 0:
        print(f"Processed {index} rows... (Successes: {success_count})")

print(f"\nFinished. Total successful extractions: {success_count}")

if features_list:
    result_df = pd.DataFrame(features_list)

    # --- הוספת עמודות ה-RAW (לפני הנרמול) ---
    print("Saving Raw Mean values...")
    # יצירת העתק של עמודות הממוצעים לעמודות חדשות עם קידומת Raw_
    for roi in ['Head', 'Body', 'Tail']:
        col_name = f'{roi}_Mean'
        if col_name in result_df.columns:
            result_df[f'Raw_{col_name}'] = result_df[col_name]

    # --- נרמול Z-Score (רק לעמודות המקוריות, לא ל-Raw) ---
    print("Normalizing data (Z-Score)...")
    cols_to_norm = [c for c in result_df.columns
                    if
                    any(x in c for x in ['Mean', 'Var', 'Skew', 'Kurt', 'Entropy', 'Contrast', 'Corr', 'Homogeneity'])
                    and 'Raw_' not in c]  # מוודא שלא מנרמלים את עמודות ה-Raw

    for col in cols_to_norm:
        if result_df[col].std() > 0:
            result_df[col] = (result_df[col] - result_df[col].mean()) / result_df[col].std()

    # סידור העמודות כך שה-Raw יופיעו בהתחלה או בסוף (אופציונלי, כאן הם פשוט מתווספים)
    result_df.to_excel(output_path, index=False)
    print(f"Saved results to: {output_path}")
else:
    print("No data extracted.")