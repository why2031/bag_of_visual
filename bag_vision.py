import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from matplotlib import gridspec

IMAGE_FOLDER = "database_images"
QUERY_IMAGE_PATH = "query.jpg"
NUM_CLUSTERS = 50
USE_SIFT = True

def load_images_from_folder(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, "*.*"))
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return image_paths, images

def extract_features(img, use_sift=True):
    if use_sift:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
    else:
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def build_dictionary(all_descriptors, num_clusters=50):
    print(f"Performing KMeans clustering with K={num_clusters} ...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(all_descriptors)
    print("KMeans finished.")
    return kmeans, kmeans.cluster_centers_

def compute_bovw_hist(descriptors, kmeans, num_clusters=50):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(num_clusters, dtype=float)
    cluster_result = kmeans.predict(descriptors)
    hist, _ = np.histogram(cluster_result, bins=range(num_clusters + 1), density=False)
    return hist.astype(float)

def compute_tf_idf(bovw_hists):
    tf = normalize(bovw_hists, norm='l1', axis=1)
    num_docs, num_words = tf.shape
    df = np.count_nonzero(bovw_hists, axis=0)
    idf = np.log((num_docs + 1) / (df + 1)) + 1
    tf_idf_hists = tf * idf
    return tf_idf_hists

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def visualize_results(query_img, db_images, db_image_paths, sorted_indices, similarities, top_k=5):
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, top_k, height_ratios=[1, 1], width_ratios=[1]*top_k)
    query_ax = plt.subplot(gs[0, :])
    query_ax.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    query_ax.set_title("Query Image", fontsize=16)
    query_ax.axis('off')
    query_pos = query_ax.get_position()
    query_center_x = (query_pos.x0 + query_pos.x1) / 2
    query_center_y = query_pos.y0
    for i in range(top_k):
        idx = sorted_indices[i]
        ax = plt.subplot(gs[1, i])
        ax.imshow(cv2.cvtColor(db_images[idx], cv2.COLOR_BGR2RGB))
        filename = os.path.basename(db_image_paths[idx])
        ax.set_title(f"Rank #{i+1}\nSimilarity: {similarities[idx]:.3f}\n{filename}", fontsize=10)
        ax.axis('off')
        similar_pos = ax.get_position()
        similar_center_x = (similar_pos.x0 + similar_pos.x1) / 2
        similar_center_y = similar_pos.y1
        line_width = max(0.5, similarities[idx] * 5)
        color = plt.cm.viridis(similarities[idx])
        line = plt.Line2D(
            [query_center_x, similar_center_x],
            [query_center_y, similar_center_y],
            transform=fig.transFigure,
            color=color,
            linewidth=line_width,
            alpha=0.7
        )
        fig.add_artist(line)
        text_x = (query_center_x + similar_center_x) / 2
        text_y = (query_center_y + similar_center_y) / 2
        plt.figtext(
            text_x, text_y,
            f"{similarities[idx]:.2f}",
            ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
            fontsize=9
        )
    plt.suptitle("Image Retrieval Visualization", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    save_filename = "retrieval_results.png"
    plt.savefig(save_filename)
    print(f"Saved retrieval visualization to {save_filename}")

    plt.show()

def draw_matching_features(query_img, db_img, query_keypoints, db_keypoints, matches, title="Feature Matching"):
    match_img = cv2.drawMatches(
        query_img, query_keypoints,
        db_img, db_keypoints,
        matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=14)
    plt.axis('off')

    safe_title = title.replace(" ", "_").replace("|", "_")
    save_filename = f"{safe_title}.png"
    plt.savefig(save_filename)
    print(f"Saved matching result to {save_filename}")

    plt.show()

def visualize_topk_matches(query_img, query_keypoints, query_desc,
                           db_images, db_keypoints_list, db_descriptors_list, db_image_paths,
                           sorted_indices, top_k=5):
    if query_desc is None or len(query_desc) == 0:
        print("No descriptors in query image; cannot visualize feature matches.")
        return
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for rank in range(top_k):
        idx = sorted_indices[rank]
        db_desc = db_descriptors_list[idx]
        db_kp = db_keypoints_list[idx]
        if db_desc is None or len(db_desc) == 0:
            print(f"Top-{rank+1} image has no descriptors. Skipping feature matching.")
            continue
        matches_knn = flann.knnMatch(query_desc, db_desc, k=2)
        good_matches = []
        for m, n in matches_knn:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        db_image = db_images[idx]
        title = f"Feature_Matching_Top{rank+1}_{os.path.basename(db_image_paths[idx])}_Matches_{len(good_matches)}"
        draw_matching_features(query_img, db_image, query_keypoints, db_kp, good_matches, title)

def main():
    db_image_paths, db_images = load_images_from_folder(IMAGE_FOLDER)
    print(f"Number of images in the database: {len(db_images)}")
    if len(db_images) == 0:
        print("No images found. Please check your database folder path.")
        return
    all_descriptors = []
    db_descriptors_list = []
    db_keypoints_list = []
    for img in db_images:
        keypoints, descriptors = extract_features(img, USE_SIFT)
        db_keypoints_list.append(keypoints)
        db_descriptors_list.append(descriptors)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    all_descriptors = np.vstack([desc for desc in db_descriptors_list if desc is not None])
    print(f"Total descriptors extracted: {all_descriptors.shape[0]}")
    kmeans_model, cluster_centers = build_dictionary(all_descriptors, NUM_CLUSTERS)
    db_bovw_hists = []
    for desc in db_descriptors_list:
        hist = compute_bovw_hist(desc, kmeans_model, NUM_CLUSTERS)
        db_bovw_hists.append(hist)
    db_bovw_hists = np.array(db_bovw_hists)
    print(f"Database BoVW histograms shape: {db_bovw_hists.shape}")
    db_tf_idf_hists = compute_tf_idf(db_bovw_hists)
    query_img = cv2.imread(QUERY_IMAGE_PATH, cv2.IMREAD_COLOR)
    if query_img is None:
        print("Failed to load query image. Please check QUERY_IMAGE_PATH.")
        return
    query_keypoints, query_desc = extract_features(query_img, USE_SIFT)
    query_hist = compute_bovw_hist(query_desc, kmeans_model, NUM_CLUSTERS).reshape(1, -1)
    num_docs = db_tf_idf_hists.shape[0]
    df = np.count_nonzero(db_bovw_hists, axis=0)
    idf = np.log((num_docs + 1) / (df + 1)) + 1
    query_tf = normalize(query_hist, norm='l1', axis=1)
    query_tf_idf = query_tf * idf
    similarities = []
    for i in range(len(db_tf_idf_hists)):
        score = cosine_similarity(db_tf_idf_hists[i], query_tf_idf[0])
        similarities.append(score)
    similarities = np.array(similarities)
    top_k = min(5, len(db_images))
    sorted_indices = np.argsort(-similarities)
    print(f"\nTop {top_k} similar images to the query:")
    for rank in range(top_k):
        idx = sorted_indices[rank]
        print(f"Rank {rank+1} | Similarity: {similarities[idx]:.4f} | Path: {db_image_paths[idx]}")
    visualize_results(query_img, db_images, db_image_paths, sorted_indices, similarities, top_k)
    if USE_SIFT and query_keypoints is not None and len(query_keypoints) > 0:
        visualize_topk_matches(
            query_img, query_keypoints, query_desc,
            db_images, db_keypoints_list, db_descriptors_list, db_image_paths,
            sorted_indices, top_k
        )

if __name__ == "__main__":
    main()
