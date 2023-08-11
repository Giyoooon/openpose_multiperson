import cv2
import time
import numpy as np
from random import randint
import argparse

image_file = "h1"
parser = argparse.ArgumentParser(description="Run keypoint detection")
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument(
    "--image_file", default="data_original/" + image_file + ".png", help="Input image"
)

args = parser.parse_args()


image1 = cv2.imread(args.image_file)
# image1 = cv2.resize(
#     image1,
#     dsize=(image1.shape[0] * (2.0 / 3), image1.shape[1] * (2.0 / 3)),
#     interpolation=cv2.INTER_CUBIC,
# )

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = [
    "Nose",
    "Neck",
    "R-Sho",
    "R-Elb",
    "R-Wr",
    "L-Sho",
    "L-Elb",
    "L-Wr",
    "R-Hip",
    "R-Knee",
    "R-Ank",
    "L-Hip",
    "L-Knee",
    "L-Ank",
    "R-Eye",
    "L-Eye",
    "R-Ear",
    "L-Ear",
]

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 0]]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [47, 48]]

colors = [
    [0, 100, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 0, 255],
    [255, 0, 0],
    [200, 200, 0],
    [255, 0, 0],
    [200, 200, 0],
    [0, 0, 0],
]
poly_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]]

color = [0, 0, 0]


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    # find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if nA != 0 and nB != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(
                        zip(
                            np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples),
                        )
                    )
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append(
                            [
                                pafA[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                                pafB[
                                    int(round(interp_coord[k][1])),
                                    int(round(interp_coord[k][0])),
                                ],
                            ]
                        )
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (
                        len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples
                    ) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(
                        valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0
                    )

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else:  # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += (
                        keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]
                    )

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = (
                        sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2])
                        + valid_pairs[k][i][2]
                    )
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


frameWidth = image1.shape[1]
frameHeight = image1.shape[0]

t = time.time()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight / frameHeight) * frameWidth)

inpBlob = cv2.dnn.blobFromImage(
    image1, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False
)
net.setInput(inpBlob)

output = net.forward()
print("Time Taken in forward pass = {}".format(time.time() - t))

detected_keypoints = []
keypoints_list = np.zeros((0, 3))
keypoint_id = 0
threshold = 0.1
interested_joints = [0, 1, 2, 3, 4, 5, 6, 7]
for part in interested_joints:
    probMap = output[0, part, :, :]
    probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
    keypoints = getKeypoints(probMap, threshold)
    print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)

# frameClone = image1.copy()
image1_height, image1_width, _ = image1.shape
frameClone = np.ones((image1_height, image1_width, 3), dtype=np.uint8) * 255


valid_pairs, invalid_pairs = getValidPairs(output)
personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
# print(personwiseKeypoints)


# 사람마다 검출한 스켈레톤 이어주기
for i in range(len(POSE_PAIRS) - 1):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
        # print(personwiseKeypoints[n][0].astype(int))
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        # dots.append([B[0], A[0]])

        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), [0, 0, 0], 10, cv2.LINE_AA)


edge_list = np.zeros((0, 3))

# 1. 모든 게임 참가자 좌우 손목 (팔꿈치) numpy array에 저장
for person in range(len(personwiseKeypoints)):
    right_wrist_index = (
        personwiseKeypoints[person][4]
        if personwiseKeypoints[person][4] != -1
        else personwiseKeypoints[person][3]
    )
    left_wrist_index = (
        personwiseKeypoints[person][7]
        if personwiseKeypoints[person][7] != -1
        else personwiseKeypoints[person][6]
    )
    edge_list = np.vstack(
        [edge_list, np.int32(keypoints_list[right_wrist_index.astype(int)])]
    )
    edge_list = np.vstack(
        [edge_list, np.int32(keypoints_list[left_wrist_index.astype(int)])]
    )

# 2. 하나씩 탐색하면서 가장 가까운 점 찾기
matched_index = set()
for index in range(len(edge_list)):
    if index in matched_index:
        continue
    matched_index.add(index)

    min_dist = np.linalg.norm(np.array([image1_height, image1_width]))
    min_index = -1
    for other_index in range(len(edge_list)):
        # 이미 연결한 손 or 같은 사람의 손 : pass
        if other_index in matched_index or index // 2 == other_index // 2:
            continue
        cur_dist = np.linalg.norm(edge_list[index] - edge_list[other_index])
        # 3. 기준 점 + 가장 가까운 점 체크
        if cur_dist < min_dist:
            min_index, min_dist = other_index, cur_dist

    if min_index == -1:
        continue
    matched_index.add(min_index)

    toDots = np.int32(edge_list[index])
    fromDots = np.int32(edge_list[min_index])
    cv2.line(
        frameClone,
        (toDots[0], toDots[1]),
        (fromDots[0], fromDots[1]),
        [0, 0, 0],
        10,
        cv2.LINE_AA,
    )

frameClone = cv2.resize(frameClone, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)

cv2.imshow("Detected Pose", frameClone)
cv2.imwrite("./data_process/" + image_file + ".png", frameClone)
cv2.waitKey(0)
