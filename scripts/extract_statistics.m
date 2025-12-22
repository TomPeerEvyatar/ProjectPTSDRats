%% === סקריפט לבניית טבלת סטטיסטיקה ===
clear; clc;

%% === הגדרות קובץ המטה דטה ===
% ודא שהנתיב הזה נכון למחשב שלך
metadataFile = "C:\Users\Tom\OneDrive - Afeka College Of Engineering\פרויקט גמר\Segmentation\תמונות מניסויים\ניסוי מנע ראשון, אפריל (בר)\clean_metadata_all_experiments.xlsx";

fprintf("בודק קובץ: %s\n", metadataFile);

if ~isfile(metadataFile)
    error("❌ קובץ המטה דטה לא נמצא: %s", metadataFile);
end

fprintf("✓ קובץ נמצא!\n\n");

% קריאת טבלת המטה דטה
meta = readtable(metadataFile, "TextType", "string");

% בדיקת עמודות (תוקן ל-Label)
requiredCols = ["Thermal_Image_Path", "Mask_mat_Path", "csv_Path", ...
                "Rat_number_unique", "Label"];
                
fprintf("בדיקת עמודות:\n");
for c = requiredCols
    if ~any(strcmp(meta.Properties.VariableNames, c))
        error("העמודה %s לא נמצאה בטבלה!", c);
    end
    fprintf("  ✓ %s\n", c);
end

n = height(meta);
fprintf("\nמספר שורות: %d\n\n", n);

results = table();
successCount = 0;
failCount = 0;

%% === לולאה על כל השורות ===
for i = 1:n
    try
        fprintf("מעבד שורה %d/%d ... ", i, n);
        
        % שליפת נתיבים מקוריים מהאקסל
        imgPath  = meta.Thermal_Image_Path(i);
        maskPath = meta.Mask_mat_Path(i);
        
        % דילוג על שורות עם ערכים חסרים
        if ismissing(imgPath) || ismissing(maskPath)
            fprintf("✗ נתיב חסר באקסל\n");
            failCount = failCount + 1;
            continue;
        end
        
        % === תיקון נתיבים (Path Correction) ===
        % הנתיבים באקסל הם של "simto", נחליף אותם ל-"Tom" אוטומטית
        % (אם שם המשתמש שלך אחר, שנה את "Tom" לשם שלך)
        imgPath = replace(imgPath, "C:\Users\simto\", "C:\Users\Tom\");
        maskPath = replace(maskPath, "C:\Users\simto\", "C:\Users\Tom\");
        
        % המרה ל-char כדי שפונקציות הקריאה יעבדו בוודאות
        imgPath = char(imgPath);
        maskPath = char(maskPath);
        
        % קריאת תמונת החום
        tempImg = readThermalImage(imgPath);
        
        % קריאת מסכה
        mask = readMask(maskPath);
        
        headMask = (mask == 1);
        bodyMask = (mask == 2);
        tailMask = (mask == 3);
        
        if ~any(headMask(:)) || ~any(bodyMask(:)) || ~any(tailMask(:))
            fprintf("✗ מסכה ריקה\n");
            failCount = failCount + 1;
            continue;
        end
        
        % פיצ'רים לכל אזור
        headFeat = computeRegionFeatures(tempImg, headMask);
        bodyFeat = computeRegionFeatures(tempImg, bodyMask);
        tailFeat = computeRegionFeatures(tempImg, tailMask);
        
        % חישובים גלובליים
        ratMask   = headMask | bodyMask | tailMask;
        ratPixels = tempImg(ratMask);
        
        Temp_range      = max(ratPixels) - min(ratPixels);
        DeltaT_HeadTail = headFeat.Mean - tailFeat.Mean;
        DeltaT_BodyTail = bodyFeat.Mean - tailFeat.Mean;
        
        % בניית שורת פלט
        row = table();
        row.Rat_number_unique = meta.Rat_number_unique(i);
        
        % Head
        row.Head_Mean         = headFeat.Mean;
        row.Head_Variance     = headFeat.Variance;
        row.Head_Skewness     = headFeat.Skewness;
        row.Head_Kurtosis     = headFeat.Kurtosis;
        row.Head_Entropy      = headFeat.Entropy;
        row.Head_Contrast     = headFeat.Contrast;
        row.Head_Correlation  = headFeat.Correlation;
        row.Head_Homogeneity  = headFeat.Homogeneity;
        
        % Body
        row.Body_Mean         = bodyFeat.Mean;
        row.Body_Variance     = bodyFeat.Variance;
        row.Body_Skewness     = bodyFeat.Skewness;
        row.Body_Kurtosis     = bodyFeat.Kurtosis;
        row.Body_Entropy      = bodyFeat.Entropy;
        row.Body_Contrast     = bodyFeat.Contrast;
        row.Body_Correlation  = bodyFeat.Correlation;
        row.Body_Homogeneity  = bodyFeat.Homogeneity;
        
        % Tail
        row.Tail_Mean         = tailFeat.Mean;
        row.Tail_Variance     = tailFeat.Variance;
        row.Tail_Skewness     = tailFeat.Skewness;
        row.Tail_Kurtosis     = tailFeat.Kurtosis;
        row.Tail_Entropy      = tailFeat.Entropy;
        row.Tail_Contrast     = tailFeat.Contrast;
        row.Tail_Correlation  = tailFeat.Correlation;
        row.Tail_Homogeneity  = tailFeat.Homogeneity;
        
        % Global
        row.Temp_range        = Temp_range;
        row.DeltaT_HeadTail   = DeltaT_HeadTail;
        row.DeltaT_BodyTail   = DeltaT_BodyTail;
        
        % Areas
        row.Head_area         = nnz(headMask);
        row.Body_area         = nnz(bodyMask);
        row.Tail_area         = nnz(tailMask);
        
        % Label
        row.Label             = meta.Label(i);
        
        results = [results; row];
        successCount = successCount + 1;
        fprintf("✓\n");
        
    catch ME
        fprintf("✗ שגיאה: %s\n", ME.message);
        failCount = failCount + 1;
    end
end

%% === כתיבה לאקסל ===
outFile = "RAT_STATISTICS.xlsx";
writetable(results, outFile);
fprintf("\n=== סיכום ===\n");
fprintf("✓ קובץ הפיצ'רים נשמר!\n");
fprintf("מיקום: %s\n", fullfile(pwd, outFile));
fprintf("הצלחות: %d/%d (%.1f%%)\n", successCount, n, 100*successCount/n);
fprintf("כשלונות: %d\n", failCount);

%% ===== פונקציות עזר =====
function tempImg = readThermalImage(pathStr)
    if ~isfile(pathStr)
        error("קובץ תמונת החום לא נמצא: %s", pathStr);
    end
    
    [~,~,ext] = fileparts(pathStr);
    switch lower(ext)
        case {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            I = imread(pathStr);
            if ndims(I) == 3
                I = rgb2gray(I);
            end
            tempImg = double(I);
        case '.mat'
            S = load(pathStr);
            fns = fieldnames(S);
            tempImg = double(S.(fns{1}));
        otherwise
            error("סיומת לא נתמכת: %s", ext);
    end
end

function mask = readMask(pathStr)
    if ~isfile(pathStr)
        error("קובץ המסכה לא נמצא: %s", pathStr);
    end
    
    [~,~,ext] = fileparts(pathStr);
    switch lower(ext)
        case {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            M = imread(pathStr);
            if ndims(M) == 3
                M = M(:,:,1);
            end
            mask = double(M);
        case '.mat'
            S = load(pathStr);
            fns = fieldnames(S);
            mask = double(S.(fns{1}));
        otherwise
            error("סיומת לא נתמכת: %s", ext);
    end
end

function feats = computeRegionFeatures(tempImg, roiMask)
    roiPixels = tempImg(roiMask);
    Ng = 64;
    
    if isempty(roiPixels)
        feats.Mean        = NaN;
        feats.Variance    = NaN;
        feats.Skewness    = NaN;
        feats.Kurtosis    = NaN;
        feats.Entropy     = NaN;
        feats.Contrast    = NaN;
        feats.Correlation = NaN;
        feats.Homogeneity = NaN;
        return;
    end
    
    % סטטיסטיקה בסיסית
    feats.Mean     = mean(roiPixels, "omitnan");
    feats.Variance = var(roiPixels, 0, "omitnan");
    
    % חישוב ידני של skewness
    mu = feats.Mean;
    sigma = sqrt(feats.Variance);
    if sigma > 0
        feats.Skewness = mean(((roiPixels - mu) / sigma).^3);
        feats.Kurtosis = mean(((roiPixels - mu) / sigma).^4);
    else
        feats.Skewness = 0;
        feats.Kurtosis = 3;
    end
    
    % אנטרופיה
    roiNorm = mat2gray(roiPixels);
    roiUint = uint8(roiNorm * (Ng - 1));
    counts = imhist(roiUint, Ng);
    p = counts / sum(counts);
    p(p == 0) = [];
    feats.Entropy = -sum(p .* log2(p));
    
    % GLCM
    props = regionprops(roiMask, "BoundingBox");
    if isempty(props)
        feats.Contrast = NaN; feats.Correlation = NaN; feats.Homogeneity = NaN;
        return;
    end
    
    bb = round(props(1).BoundingBox);
    r1 = max(bb(2), 1);
    c1 = max(bb(1), 1);
    r2 = min(r1 + bb(4) - 1, size(tempImg,1));
    c2 = min(c1 + bb(3) - 1, size(tempImg,2));
    
    tempNorm = mat2gray(tempImg);
    tempUint = uint8(tempNorm * (Ng - 1));
    
    % חיתוך האזור הרלוונטי (Bounding Box)
    subImg  = tempUint(r1:r2, c1:c2);
    
    % חישוב GLCM
    glcm = graycomatrix(subImg, 'NumLevels', Ng, 'Symmetric', true);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Homogeneity'});
    
    feats.Contrast    = mean(stats.Contrast);
    feats.Correlation = mean(stats.Correlation);
    feats.Homogeneity = mean(stats.Homogeneity);
end
