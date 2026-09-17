#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ADPC: Adaptive Density Peaks Clustering Based on Natural Nearest Neighbors
===========================================================================

Overview
--------
This module implements **ADPC**, a density-peak-based clustering algorithm
that is free from manually specified hyper-parameters (e.g., the cutoff
distance ``dc`` or a fixed neighborhood size ``k``). Instead, ADPC derives
the neighborhood scale in a data-driven manner through *natural neighbor*
search, and then combines local density peaks, representative-point
propagation, hierarchical merging (guided by shared nearest-neighbor
similarity) and a fuzzy-membership-based label propagation strategy to
produce the final clustering result. The overall pipeline consists of the
following stages:

    1. Distance matrix construction (feature normalization + pairwise
       Euclidean distance).
    2. Natural neighbor search, which adaptively determines the
       neighborhood size ``k`` and the search radius ``lamda``.
    3. Local density and relative distance estimation for every sample.
    4. Local density peak (LDP) detection and representative-point
       assignment, followed by representative-chain adjustment.
    5. Automatic estimation of the cluster number via shared-nearest-
       neighbor-weighted hierarchical clustering of local density peaks.
    6. Cluster center selection and initial label assignment.
    7. Fuzzy-membership-based label propagation for samples that remain
       unlabeled after the initial assignment.
    8. Final label completion for any residual isolated samples.

Author Information
-------------------
    Chinese Name : 严欢 (YAN Huan)
    English Name : hwanyan
    Email        : yan-huan@snnu.edu.cn
    ORCID        : https://orcid.org/0009-0000-0016-6324
    Affiliations :
        1. Tencent Cloud Computing (Chongqing) Co., Ltd.
        2. Shaanxi Normal University

How to Cite
-----------
If this implementation is useful for your research, please cite the
corresponding paper:

    Xie, J., Yan, H., Wang, M., Grant, P. W., & Pedrycz, W. (2026).
    Automated clustering with density peaks. Manuscript submitted to
    Pattern Recognition.

    BibTeX::

        @article{xie2026adpc,
            title   = {Automated clustering with density peaks},
            author  = {Xie, Juanying and Yan, Huan and Wang, Mingzhao and
                        Grant, Philip W. and Pedrycz, Witold},
            journal = {Pattern Recognition},
            year    = {2026},
            note    = {Manuscript submitted for publication}
        }

License
-------
Released under the MIT License. See the ``LICENSE`` file in this
repository for full terms of use.

Dependencies
------------
- numpy
- scikit-learn (``sklearn.preprocessing.MinMaxScaler``)
- scipy (``scipy.cluster.hierarchy``)
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster import hierarchy
from queue import Queue
from collections import Counter


class ADPC:
    """Adaptive Density Peaks Clustering based on Natural Nearest Neighbors.

    The class exposes a scikit-learn-like workflow: instantiate with the
    feature matrix, then call :meth:`fit` to run the full clustering
    pipeline. After fitting, cluster assignments are available via
    :attr:`Labels` and the indices of the discovered cluster centers via
    :attr:`ClusterCenters`.

    Attributes
    ----------
    X : numpy.ndarray of shape (Size, FN)
        Feature matrix of the dataset. Overwritten in-place with its
        min-max normalized version once :meth:`fit` is executed.
    Size : int
        Number of samples in the dataset.
    FN : int
        Number of features (dimensionality) of the dataset.
    DM : numpy.ndarray of shape (Size, Size)
        Pairwise Euclidean distance matrix.
    RNNCList : numpy.ndarray of shape (Size,)
        Reverse (natural) nearest-neighbor count for each sample.
    RNNMatrix : list of list of int
        Reverse nearest-neighbor index list for each sample.
    lamda : int
        Natural neighbor characteristic value, i.e., the search radius
        (number of neighbor hops) at which the natural neighbor search
        stabilizes.
    k : int
        Natural neighbor number, i.e., the adaptively determined
        neighborhood size used throughout the algorithm.
    Density : numpy.ndarray of shape (Size,)
        Local density of each sample.
    RD : numpy.ndarray of shape (Size,)
        Relative distance of each sample.
    Object : numpy.ndarray of shape (Size,)
        For each sample, the index of its "parent" sample used when
        computing the relative distance (the nearest higher-density
        sample).
    DV : numpy.ndarray of shape (Size,)
        Decision value of each sample, defined as ``Density * RD``.
    CN : int
        Estimated number of clusters. Initialized to ``-1`` and set by
        :meth:`__getClusterNumber`.
    ClusterCenters : list of int
        Indices of the samples selected as cluster centers.
    Labels : numpy.ndarray of shape (Size,)
        Predicted cluster label for each sample (``0`` denotes an
        as-yet-unassigned sample; final clusters are labeled from ``1``).
    """

    def __init__(self, X):
        """Initialize the ADPC estimator.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The feature matrix of the dataset to be clustered.
        """
        # Feature matrix
        self.X = X
        # The number of samples included in the dataset
        self.Size = len(X)
        # The number of features in the dataset
        self.FN = X.shape[1]
        # Distance matrix
        self.DM = []
        # Reverse (natural) nearest-neighbor count list
        self.RNNCList = []
        # Reverse nearest-neighbor index matrix
        self.RNNMatrix = []
        # Natural neighbor characteristic value (search radius)
        self.lamda = 0
        # Natural neighbor number (adaptive neighborhood size)
        self.k = 0
        # Local density vector
        self.Density = []
        # Relative distance vector
        self.RD = np.zeros(self.Size, dtype=float)
        # Object (parent) vector used for relative distance computation
        self.Object = np.zeros(self.Size, dtype=int)
        # Decision value vector
        self.DV = []
        # Number of clusters (estimated automatically)
        self.CN = -1
        # Indices of the selected cluster centers
        self.ClusterCenters = []
        # Predicted cluster labels
        self.Labels = np.zeros(self.Size, dtype=int)

    def __Normalization(self, X):
        """Standardize the feature matrix via min-max scaling.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Raw feature matrix.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            Feature matrix with every column linearly rescaled to the
            ``[0, 1]`` range.
        """
        # 1. Instantiate the transformer (feature_range defines the target
        #    normalization interval, i.e., [minimum, maximum])
        transfer = MinMaxScaler(feature_range=(0, 1))
        # 2. Fit and transform the input features
        return transfer.fit_transform(X)

    def __getDistanceMatrix(self):
        """Compute the pairwise distance matrix of the dataset.

        The feature matrix is first min-max normalized, and the pairwise
        Euclidean distance is then computed for every pair of samples.
        (If discrete attributes were present, a mixed-distance metric
        would be required here instead.)

        Returns
        -------
        None
            Populates ``self.DM`` in place.
        """
        # Initialize the distance matrix
        self.DM = np.zeros((self.Size, self.Size), dtype=float)
        # Normalize the input data
        self.X = self.__Normalization(self.X)
        # Compute the distance matrix (based on Euclidean distance)
        for i in range(self.Size - 1):
            for j in range(i + 1, self.Size):
                dis = np.sum(np.power(self.X[i] - self.X[j], 2)) ** 0.5
                self.DM[i][j] = self.DM[j][i] = dis

    def __getNearstNeighborInfomation(self):
        """Derive nearest-neighbor ordering information from the distance matrix.

        Builds, for every sample:
            1. The neighbor index matrix, i.e., the indices of all other
               samples sorted in ascending order of distance.
            2. The corresponding sorted distance matrix.

        Returns
        -------
        None
            Populates ``self.__NearstNeighbors`` and
            ``self.__NearstNeighborsDis`` in place.
        """
        self.__NearstNeighbors = np.argsort(self.DM)[:, 1:]
        self.__NearstNeighborsDis = np.sort(self.DM)[:, 1:]

    def __NaNSearch(self):
        """Perform Natural Neighbor search to adaptively determine ``k`` and ``lamda``.

        The neighbor order is expanded step by step (starting from the
        1st nearest neighbor). At each step, every sample's reverse
        nearest-neighbor count is accumulated, and the search stops once
        the number of samples having zero reverse neighbors stabilizes
        (i.e., stops decreasing further). This yields:
            - ``self.lamda``: the natural neighbor characteristic value
              (number of expansion steps performed), used later as the
              search range for shared-nearest-neighbor similarity.
            - ``self.k``: the natural neighbor number, taken as the
              maximum reverse nearest-neighbor count across all samples,
              used throughout the algorithm as the adaptive neighborhood
              size.

        Returns
        -------
        None
            Populates ``self.lamda``, ``self.k``, ``self.RNNCList`` and
            ``self.RNNMatrix`` in place.
        """
        # Initialize the natural neighbor expansion step counter
        NNC = 1
        # Initialize the count of samples having no reverse neighbors
        RNNC = self.Size
        # Initialize the matrix recording the reverse nearest neighbors of each sample
        for index in range(self.Size):
            self.RNNMatrix.append([])
        self.RNNCList = np.zeros(self.Size, dtype=int)
        while NNC < self.Size:
            for i in range(self.Size):
                # Get the NNC-th nearest neighbor r of sample i
                r = self.__NearstNeighbors[i, NNC - 1]
                # Increment the reverse nearest-neighbor count of sample r
                self.RNNCList[r] += 1
                # Record sample i as a reverse nearest neighbor of sample r
                self.RNNMatrix[r].append(i)
            # Count the current number of samples without reverse neighbors
            RNNC_new = np.sum(self.RNNCList == 0)
            # Stop expanding once this count no longer changes (stabilizes)
            if RNNC_new == RNNC:
                break
            else:
                RNNC = RNNC_new
            NNC += 1
        # Upon convergence, NNC records the natural neighbor search range
        self.lamda = NNC
        # The maximum reverse nearest-neighbor count is taken as the
        # natural neighbor number shared by all samples
        self.k = np.max(self.RNNCList)

    def __getDensity(self):
        """Compute the local density of every sample.

        The local density of a sample is defined as the negative
        exponential of the mean distance to its ``k`` natural nearest
        neighbors, so that samples in denser regions (smaller average
        neighbor distance) obtain a higher density value.

        Returns
        -------
        None
            Populates ``self.Density`` in place.
        """
        self.Density = np.exp(-np.mean(self.__NearstNeighborsDis[:, :self.k], axis=1))

    def __getRelativeDistance(self):
        """Compute the relative distance and parent (object) sample for every sample.

        Samples are processed in descending order of local density. The
        globally densest sample (the eventual cluster center) is assigned
        the maximum pairwise distance in the dataset as its relative
        distance. Every other sample's relative distance is defined as
        its minimum distance to any sample with strictly higher density
        that has already been processed; the corresponding nearest
        higher-density sample is recorded as its "object" (parent).

        Returns
        -------
        None
            Populates ``self.RD`` and ``self.Object`` in place.
        """
        # Sort samples by density in descending order
        sortDensityIndex = np.argsort(self.Density)[::-1]
        # The sample with the highest density is assigned the maximum
        # pairwise distance as its relative distance (it is necessarily
        # a cluster center candidate)
        self.RD[sortDensityIndex[0]] = np.max(self.DM)
        # Compute the relative distance and object sample for all others
        for i in range(1, self.Size):
            tempIndex = sortDensityIndex[i]
            self.Object[tempIndex] = sortDensityIndex[
                np.argmin(self.DM[tempIndex][sortDensityIndex[:i]])]
            self.RD[tempIndex] = self.DM[tempIndex][self.Object[tempIndex]]

    def __getLocalDensityPeaks(self):
        """Determine each sample's representative and detect local density peaks (LDP).

        For every sample, its representative is defined as the highest
        density sample among its ``k`` natural nearest neighbors, but
        only if that neighbor's density exceeds the sample's own
        density; otherwise the sample is its own representative and is
        therefore marked as a local density peak.

        Returns
        -------
        None
            Populates ``self.Rep`` (representative index for every
            sample) and ``self.LDP`` (indices of local density peaks)
            in place.
        """
        self.Rep = []
        self.LDP = []
        for i in range(self.Size):
            # Retrieve the neighbor with the highest density among the
            # k-nearest neighbors of sample i
            maxDensityNN = self.__NearstNeighbors[i, np.argmax(self.Density[self.__NearstNeighbors[i, :self.k]])]
            # If that neighbor's density exceeds sample i's density,
            # it becomes the representative of sample i
            if self.Density[maxDensityNN] > self.Density[i]:
                self.Rep.append(maxDensityNN)
            # Otherwise, sample i represents itself and is a local density peak
            else:
                self.Rep.append(i)
                self.LDP.append(i)
        self.Rep = np.array(self.Rep, dtype=int)
        self.LDP = np.array(self.LDP, dtype=int)

    def __adjustRepresentative(self):
        """Resolve representative chains so that every sample points directly to an LDP.

        The raw representative assigned in :meth:`__getLocalDensityPeaks`
        may form a chain of representatives that does not terminate at a
        local density peak in a single hop. This method follows each such
        chain until a local density peak is reached, and re-points every
        sample along the chain (as well as all samples that were
        transitively grouped under it) directly to that peak.

        Returns
        -------
        None
            Updates ``self.Rep`` in place so that every sample's
            representative is a local density peak.
        """
        Reps = np.unique(self.Rep)
        ClusterDict = {}
        for rep in Reps:
            ClusterDict[int(rep)] = (np.arange(self.Size)[self.Rep == rep]).tolist()

        for rep in Reps:
            chain = []
            while rep not in self.LDP:
                chain.append(rep)
                rep = self.Rep[rep]
            for r in chain:
                if r not in ClusterDict:
                    continue
                for s in ClusterDict[r]:
                    self.Rep[s] = rep
                    ClusterDict[rep].append(s)
                self.Rep[r] = rep
                ClusterDict[rep].append(r)
                del ClusterDict[r]

    def __getFREs(self, Thrs, NodeLength=100):
        """Compute the frequency rank of each merge threshold within a discretized grid.

        The input array ``Thrs`` (typically the sequence of hierarchical
        merge distances) is discretized into ``NodeLength`` equally
        spaced nodes spanning ``[Thrs[0], Thrs[-1]]``. Scanning from the
        smallest to the largest value, this method counts, for each
        element of ``Thrs``, how many of the remaining (not-yet-passed)
        elements are associated with it, effectively producing a
        descending "frequency rank" sequence used to identify the most
        stable cluster-number candidate.

        Parameters
        ----------
        Thrs : numpy.ndarray
            Sorted array of merge thresholds (e.g., the third column of a
            SciPy linkage matrix).
        NodeLength : int, optional
            Number of discretization nodes spanning the value range of
            ``Thrs``. Defaults to ``100``.

        Returns
        -------
        numpy.ndarray
            Frequency rank associated with each element of ``Thrs``.
        """
        Nodes = np.linspace(Thrs[0], Thrs[-1], NodeLength)
        p, CN = 0, len(Thrs) + 1
        FREs = []
        for thr in Thrs:
            while p < NodeLength and Nodes[p] <= thr:
                FREs.append(CN)
                p += 1
            CN -= 1
        return np.array(FREs)

    def __getClusterNumber(self):
        """Automatically estimate the number of clusters from the local density peaks.

        The procedure works as follows:
            1. Group samples by their representative local density peak.
            2. Compute the pairwise distance between every pair of local
               density peaks.
            3. Compute the shared nearest-neighbor (SNN) count between
               every pair of local-density-peak clusters, based on the
               union of each cluster's samples' ``lamda``-nearest
               neighbors.
            4. Combine peak distance and SNN similarity into an SNN-
               weighted distance (peaks sharing more neighbors are
               effectively "pulled closer together"), then perform
               agglomerative hierarchical clustering on the resulting
               condensed distance vector.
            5. Discretize the resulting merge thresholds and select the
               most frequently occurring frequency rank as the final
               estimated cluster number.

        Returns
        -------
        None
            Populates ``self.CN`` in place.
        """
        # Build a dictionary grouping samples by their local density peak
        ClusterDict = {}
        SCN = len(self.LDP)
        for i in range(1, SCN + 1):
            ClusterDict[i] = (np.arange(self.Size)[self.Rep == self.LDP[i - 1]]).tolist()

        # Compute the pairwise distance between local density peaks
        LDPDM = np.zeros((SCN, SCN), dtype=float)
        for ldp_i in range(SCN - 1):
            for ldp_j in range(ldp_i + 1, SCN):
                i, j = self.LDP[ldp_i], self.LDP[ldp_j]
                LDPDM[ldp_i][ldp_j] = self.DM[i][j]

        # Compute the lamda-nearest-neighbor union for the samples of each local density peak
        AugCluster = {}
        for cluster in ClusterDict:
            tmp = []
            for p in ClusterDict[cluster]:
                for n in self.__NearstNeighbors[p][:self.lamda]:
                    tmp.append(int(n))
            AugCluster[cluster] = list(set(tmp))
        SNNM = np.zeros((SCN, SCN), dtype=int)
        for ldp_i in range(SCN - 1):
            for ldp_j in range(ldp_i + 1, SCN):
                SNN = len(set(AugCluster[ldp_i + 1]) & set(AugCluster[ldp_j + 1]))
                SNNM[ldp_i][ldp_j] = SNN

        # Assemble the SNN-weighted condensed distance vector (upper triangle)
        ytdist = []
        maxSNN = np.max(SNNM)
        for ldp_i in range(SCN - 1):
            for ldp_j in range(ldp_i + 1, SCN):
                if SNNM[ldp_i][ldp_j] != 0:
                    yt = LDPDM[ldp_i][ldp_j] * np.exp(-SNNM[ldp_i][ldp_j] / maxSNN)
                else:
                    yt = LDPDM[ldp_i][ldp_j]
                ytdist.append(yt)

        # Perform agglomerative hierarchical clustering on the local density peaks
        Z = hierarchy.linkage(ytdist)
        Thrs = Z[:, 2]
        FREs = self.__getFREs(Thrs)
        FreCount = Counter(FREs)
        self.CN = max(FreCount, key=FreCount.get)

    def __getDecisionValue(self):
        """Compute the decision value used for cluster-center candidate ranking.

        The decision value of a sample is defined as the product of its
        local density and relative distance; samples with a high
        decision value are strong cluster-center candidates.

        Returns
        -------
        None
            Populates ``self.RD`` (via :meth:`__getRelativeDistance`) and
            ``self.DV`` in place.
        """
        self.__getRelativeDistance()
        self.DV = np.multiply(self.Density, self.RD)

    def __getClusterCenters(self):
        """Select cluster centers among local density peaks and perform initial label assignment.

        Cluster centers are selected iteratively from the local density
        peaks (LDPs), ranked by a peak-specific decision value (peak
        density times relative distance). After each center is chosen,
        its ``k``-nearest neighbors are directly labeled with the new
        cluster id (together with their own reverse nearest neighbors),
        and any LDP reachable from the center through a chain of
        mutually close, non-outlier neighbors is disqualified from
        further candidacy (breadth-first propagation via a queue). If
        the desired number of clusters (``self.CN``) cannot be reached
        using LDP candidates alone, the remaining centers are chosen
        from the overall highest decision-value samples.

        Returns
        -------
        None
            Populates ``self.ClusterCenters``, ``self.Labels`` and
            ``self.Outliers`` in place.
        """
        LDPDensitys = []
        for ldp in self.LDP:
            LDPDensitys.append(self.RNNCList[ldp] / np.sum(self.__NearstNeighborsDis[ldp, :self.RNNCList[ldp]]))
        relativeDistance = self.RD[self.LDP]
        DecisionValue = np.multiply(LDPDensitys, relativeDistance)
        sortedLDP = self.LDP[np.argsort(DecisionValue)][::-1]
        numLDP = len(self.LDP)
        # Candidate eligibility flag for each sample during cluster-center selection
        flagLDP = np.zeros(self.Size, dtype=int)

        # Identify outliers based on the k-th nearest-neighbor distance
        KthNNDistance = self.__NearstNeighborsDis[:, self.k - 1]
        OutlierThreshold = np.mean(KthNNDistance)
        self.Outliers = np.arange(self.Size)[KthNNDistance > OutlierThreshold]

        CN = 1
        while CN <= self.CN:
            # Select the eligible candidate LDP with the highest decision value as the next cluster center
            p = 0
            while p < numLDP and flagLDP[sortedLDP[p]] != 0:
                p += 1
            if p == numLDP:
                break
            center = sortedLDP[p]
            self.ClusterCenters.append(center)
            self.Labels[center] = CN
            flagLDP[sortedLDP[p]] = 1

            # Assign labels to the center's k-nearest neighbors (and their reverse neighbors)
            for n in self.__NearstNeighbors[center, :self.k]:
                if self.Labels[n] == 0:
                    self.Labels[n] = CN
                    for rn in self.RNNMatrix[n]:
                        self.Labels[rn] = CN

            # Disqualify LDPs transitively reachable from the center via close, non-outlier neighbors
            q = Queue()
            q.put(center)
            while not q.empty():
                head = q.get()
                radius = np.mean(self.__NearstNeighborsDis[head, :self.k])
                for n in self.__NearstNeighbors[head, :self.k]:
                    if flagLDP[n] == 0 and self.DM[head][n] <= radius and n not in self.Outliers:
                        flagLDP[n] = 1
                        q.put(n)
            CN += 1

        if CN <= self.CN:
            DV = np.multiply(self.Density, self.RD)
            sortedIndex = np.argsort(DV)
            p = 0
            while CN <= self.CN and p < self.Size:
                center = sortedIndex[p]
                if flagLDP[center] == 0:
                    self.Labels[center] = CN
                    flagLDP[center] = 1
                    self.ClusterCenters.append(center)

                    # Assign labels to the center's k-nearest neighbors (and their reverse neighbors)
                    for n in self.__NearstNeighbors[center, :self.k]:
                        if self.Labels[n] == 0:
                            self.Labels[n] = CN
                            for rn in self.RNNMatrix[n]:
                                self.Labels[rn] = CN

                    CN += 1

                p += 1

    def __LabelPropagated(self):
        """Label allocation strategy 2: fuzzy-membership propagation for unassigned samples.

        For every sample that remains unlabeled after
        :meth:`__getClusterCenters`, a weighted membership degree to each
        existing cluster is computed from its labeled ``k``-nearest
        neighbors, using squared sample similarity (``1 / (1 + distance)``)
        normalized by each neighbor's total k-nearest-neighbor similarity.
        Samples are then greedily assigned, in decreasing order of their
        maximum membership degree, to the cluster achieving that maximum;
        each assignment triggers an incremental update of the membership
        degrees of its still-unlabeled neighbors.

        Returns
        -------
        None
            Updates ``self.Labels`` in place for all samples that can be
            reached, directly or transitively, from an already labeled
            neighbor.
        """
        # Compute the sample similarity matrix
        self.SimilarMatrix = 1 / (1 + self.DM)
        # Compute the total k-nearest-neighbor similarity of each sample
        # (used to normalize the weighted membership degree)
        KNNSimilarSumVector = np.sum(1 / (1 + self.__NearstNeighborsDis[:, :self.k]), 1)
        # Define the identification (membership) matrix
        # (dictionary keyed by sample index; each value is a vector of
        # membership degrees over the k-nearest-neighbor classification)
        identifyMatrix = {}
        # Initialize the identification matrix (h * C):
        # compute the membership degree of each unassigned sample with respect to every cluster Ci
        for index in range(self.Size):
            # Skip samples that are already labeled
            if self.Labels[index] > 0:
                continue
            # Define the k-nearest-neighbor classification (membership) vector for this sample
            # Note: since cluster ids start from 1 while array indices start
            # from 0, the membership vector needs one extra column (+1)
            neighborClassVector = np.zeros(self.CN + 1, dtype=float)
            # Compute the membership degree of the current sample with
            # respect to each cluster (by traversing its neighbors)
            for neighbor in self.__NearstNeighbors[index, :self.k]:
                if self.Labels[neighbor] > 0:
                    neighborClassVector[self.Labels[neighbor]] += self.SimilarMatrix[index][neighbor]**2 / KNNSimilarSumVector[neighbor]
            # Store the computed membership vector in the identification matrix
            identifyMatrix[index] = neighborClassVector
        # Build the identification (decision) vector: extract the maximum
        # membership degree of each sample from the identification matrix.
        # Structure: key -> sample index, value -> [membership degree, cluster id]
        identifyVector = {}
        for index in identifyMatrix:
            # Extract the k-nearest-neighbor classification vector of the current sample
            npary = np.array(identifyMatrix[index])
            # Determine the maximum membership degree and its associated cluster id
            identifyVector[index] = [np.max(npary), np.argmax(npary)]
        # Assign labels iteratively based on the identification vector
        while len(identifyVector) > 0:
            # Select the sample with the highest membership degree
            object = max(identifyVector, key=identifyVector.get)
            # If the selected sample's maximum membership degree is zero,
            # it means none of its neighbors are labeled yet, i.e., the
            # sample belongs to an extremely isolated small cluster;
            # terminate the propagation loop
            if identifyVector[object][0] == 0:
                break
            # Assign the sample to the cluster of its maximum membership degree
            self.Labels[object] = identifyVector[object][1]
            # Update the membership degree of any unassigned neighbor of
            # this sample, since it is now newly labeled.
            # Note: this update follows the original FKNN-DPC assumption
            # that the k-nearest-neighbor relation is treated as mutual,
            # i.e., if sample x is a nearest neighbor of sample y, then y
            # is also treated as a nearest neighbor of x for this update.
            for neighboor in self.__NearstNeighbors[object, :self.k]:
                if self.Labels[neighboor] == 0:
                    # Update the identification matrix
                    identifyMatrix[neighboor][identifyVector[object][1]] += self.SimilarMatrix[neighboor][object]**2 / KNNSimilarSumVector[neighboor]
                    # Update the identification vector
                    # (note: entries already removed from the identification
                    # vector must not be updated again, to avoid an infinite loop)
                    identifyVector[neighboor] = [np.max(identifyMatrix[neighboor]), np.argmax(identifyMatrix[neighboor])]
            # Remove the just-labeled sample from the identification matrix and vector
            identifyMatrix.pop(object)
            identifyVector.pop(object)

    def __LabeledFinally(self):
        """Final label completion pass for any remaining unlabeled (isolated) samples.

        A small number of extremely isolated samples may still remain
        unlabeled after :meth:`__LabelPropagated` due to mutual exclusion
        among unlabeled neighbors that could not be broken. Each such
        sample is assigned the label of its nearest already-labeled
        neighbor.

        Returns
        -------
        None
            Finalizes ``self.Labels`` in place and converts
            ``self.ClusterCenters`` into a NumPy array.
        """
        for index in range(self.Size):
            if self.Labels[index] == 0:
                # Retrieve the nearest-neighbor ordering of the current sample
                NearstNeighbors = self.__NearstNeighbors[index]
                # Traverse neighbors in ascending distance order and adopt
                # the label of the first already-labeled neighbor found
                for neighbor in NearstNeighbors:
                    if self.Labels[neighbor] > 0:
                        self.Labels[index] = self.Labels[neighbor]
                        break
        self.ClusterCenters = np.array(self.ClusterCenters)

    def fit(self):
        """Run the complete ADPC clustering pipeline.

        Executing this method sequentially performs distance-matrix
        construction, natural neighbor search, local density and
        relative-distance estimation, local density peak detection,
        representative-chain adjustment, automatic cluster-number
        estimation, cluster-center selection with initial label
        assignment, fuzzy-membership label propagation, and final label
        completion.

        Returns
        -------
        None
            After calling this method, the fitted results are available
            via ``self.Labels`` (predicted cluster label per sample) and
            ``self.ClusterCenters`` (indices of the selected cluster
            centers).
        """
        self.__getDistanceMatrix()
        self.__getNearstNeighborInfomation()
        self.__NaNSearch()
        self.__getDensity()
        self.__getLocalDensityPeaks()
        self.__adjustRepresentative()
        self.__getClusterNumber()
        self.__getDecisionValue()
        self.__getClusterCenters()
        self.__LabelPropagated()
        self.__LabeledFinally()
