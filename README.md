# bag_of_visual

This project applies a Bag of Visual Words (BoVW) approach for image similarity and retrieval. The data is taken from the [Caltech 256](https://www.kaggle.com/datasets/jessicali9530/caltech256/data) dataset on Kaggle; some categories are placed in the `database_images` folder, and one image (`query.jpg`) is used as the query.

**Key steps:**

1. Extract local features (SIFT or ORB) from each image.
2. Build a visual vocabulary (e.g., 50 clusters) using KMeans on all descriptors.
3. Convert each image into a histogram of visual words, optionally applying TF-IDF weighting.
4. Compute cosine similarity between the query and database images, then display a ranked list.
5. Provide visualizations showing the top results and their feature matches with the query image.

By running the script, you can see the console output showing the most similar images and a pop-up window illustrating how the query image’s features match those in the returned results.



**Result：**

Number of images in the database: 95
Total descriptors extracted: 56123
Performing KMeans clustering with K=50 ...
KMeans finished.
Database BoVW histograms shape: (95, 50)

Top 5 similar images to the query:
Rank 1 | Similarity: 0.8407 | Path: database_images\001_0012.jpg
Rank 2 | Similarity: 0.8183 | Path: database_images\001_0085.jpg
Rank 3 | Similarity: 0.7850 | Path: database_images\001_0077.jpg
Rank 4 | Similarity: 0.7756 | Path: database_images\001_0021.jpg
Rank 5 | Similarity: 0.7679 | Path: database_images\014_0019.jpg

<br/> <div align="center"> <img src="https://img.picui.cn/free/2025/03/19/67dae7ee57924.png" alt="Query and retrieved results" width="600"/> </div> <br/>
<div align="center"> <img src="https://img.picui.cn/free/2025/03/19/67dae7eef18ef.png" alt="Feature matching visualization" width="600"/> </div>




