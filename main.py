import numpy as np
from wand.image import Image
import matplotlib.pyplot as plt
import cv2 as cv
import os


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path

def is_identical(img1_path:str, img2_path:str, save=False, display=False, fuzz_thresh=0.25, thresh_in_pixels=10):
    img1_name = os.path.splitext(os.path.basename(img1_path))[0]
    img2_name = os.path.splitext(os.path.basename(img2_path))[0]

    with Image(filename=img1_path) as bimg:
        with Image(filename=img2_path) as fimg:
            bimg.fuzz = fuzz_thresh * bimg.quantum_range
            bimg.artifacts['compare:highlight-color'] = 'red'
            bimg.artifacts['compare:lowlight-color'] = 'black'
            diff_img, _ =  bimg.compare(fimg, 'fuzz')

            with diff_img:
                if save:
                    diff_img.save(filename=f'images/{img1_name}_{img2_name}.jpg')
                if display:
                    display(diff_img)

                dimg = np.array(diff_img)[:, :, :3]

            artifacts_cnt = np.sum(dimg > 0)
            print(f"Different pixels: {artifacts_cnt}, maximum outliers allowed: {thresh_in_pixels}, fuzz thresh: {fuzz_thresh}")
            if artifacts_cnt > thresh_in_pixels:
                print(f"{img1_name} and {img2_name} are visually different")
                result = False
            else:
                result = True
    
    return result, dimg

def altering(img):
    row_range = np.random.randint(5, 20)
    col_range = np.random.randint(5, 20)

    offset_row = np.random.randint(0, img.shape[0] - row_range - 1)
    offset_col = np.random.randint(0, img.shape[1] - col_range - 1)

    altered = img.copy()
    altered[offset_row: offset_row + row_range, offset_col: offset_col + col_range, :] =  np.random.randint(low = 0, high = 255, size = (row_range, col_range, 3))  

    return altered

def debug_function(img_path, results=[], expected=[]):
    name = os.path.splitext(os.path.basename(img_path))[0]

    original = cv.imread(img_path)
    compressed_path = f"images/{name}_JPEG2000_0.jpg"
    cv.imwrite(compressed_path, original, [cv.IMWRITE_JPEG2000_COMPRESSION_X1000, 0])

    compressed = cv.imread(compressed_path)
    diff_compressed = original - compressed
    diff_compr_result = "identical" if np.all(diff_compressed == 0) else "different"
    print(f"Compressed and Original are binary {diff_compr_result}")

    # binary_diff = original-compressed
    # cv.imshow("diff", binary_diff)
    # cv.waitKey()

    altered = altering(original)
    altered_path = f"images/{name}_altered.jpg"
    cv.imwrite(altered_path, altered)
    
    altered_compressed = altering(compressed)
    altered_compressed_path = f"images/{name}_JPEG2000_0_altered.jpg"
    cv.imwrite(altered_compressed_path, altered_compressed)

    res1, _ = is_identical(img1_path=img_path, img2_path=img_path, save=True) # Should be 0 diffs
    res2, dimg1 = is_identical(img1_path=img_path, img2_path=compressed_path, save=True) # Should be greater than zero diff, but visually ok (diff under thresh)
    res3, dimg2 = is_identical(img1_path=img_path, img2_path=altered_path, save=True) # Should fail the comparison test
    res4, dimg3 = is_identical(img1_path=img_path, img2_path=altered_compressed_path, save=True) # Should fail the comparison test

    visuals = [compressed, altered]
    diff_maps = [dimg1, dimg2]
    titles = [res2, res3]

    fig = plt.figure(figsize=(22, 15))
    plt.axis('off')
    # ax = fig.add_subplot(111)
    # ax.imshow(original)
    for idx, (img1, img2, res) in enumerate(zip(visuals, diff_maps, titles)):
        ax = fig.add_subplot(2, 2, 2*idx+1)
        ax.imshow(img1)

        ax = fig.add_subplot(2, 2, 2*idx+2)
        ax.imshow(img2)

        if not res: # Failed
            ax.set_title("Failed")
        else:
            ax.set_title("Passed")

    path = f"images/results/{name}_results.png"
    path = uniquify(path)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
        

    results.extend([res1, res2, res3, res4])
    expected.extend([True, True, False, False])


results = []
expected = []

for _ in range(10): # loop for randomization of different image artifacts
    for subdir, dirs, files in os.walk(os.path.join(os.getcwd(),"images/originals")):
        for file in files:
            path = os.path.join(subdir, file)
            debug_function(os.path.join(subdir, file), results=results, expected=expected)

FP = 0 # RESULTS are non identical images and images are identical (False Alarm)  
FN = 0 # RESULTS are identical images and images are visualy different  

for idx,_ in enumerate(results):
    print(idx+1, end=", ")
    if results[idx] and not expected[idx]:
        FN += 1
        print("FN")
    if not results[idx] and expected[idx] :
        FP += 1
        print("FP")
    else:
        print("Passed")

if results:
    FP = FP / len(results)
    FN = FN / len(results)
    print(F"FP={FP*100}%, FN={FN*100}%")


else:
    print("No results")

