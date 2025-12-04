# Project2
    ====================================================
NEURAL LSH - ΑΝΑΖΗΤΗΣΗ ΚΟΝΤΙΝΩΝ ΓΕΙΤΟΝΩΝ ΜΕ ΝΕΥΡΩΝΙΚΑ ΔΙΚΤΥΑ
    ====================================================

ΠΕΡΙΓΡΑΦΗ

Αυτό το έργο υλοποιεί την τεχνική Neural Locality-Sensitive Hashing
(Neural LSH) για αποδοτική αναζήτηση κοντινών γειτόνων σε υψηλής
διάστασης δεδομένα. Η μέθοδος μαθαίνει έξυπνες διαμερίσεις του χώρου
χρησιμοποιώντας διαμέριση γράφων και νευρονικά δίκτυα.

ΚΑΤΑΛΟΓΟΣ ΑΡΧΕΙΩΝ PYTHON:

**data_reader.py**  Διαβάζει αρχεία MNIST και SIFT 
Επιστρέφει NumPy arrays

**nlsh_build.py**   ΚΑΤΑΣΚΕΥΗ ευρετηρίου Neural LSH
Αποθηκεύει: model.pth (το νευρονικό δίκτυο) και index.pkl (δεδομένα)

**nlsh_search.py**  ΑΝΑΖΗΤΗΣΗ με χρήση του Neural LSH ευρετηρίου

ΟΔΗΓΙΕΣ ΕΓΚΑΤΑΣΤΑΣΗΣ ΕΞΑΡΤΗΣΕΩΝ:
Εγκαταστήστε τις απαιτούμενες βιβλιοθήκες:
**pip install -r requirements.txt**


ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ - ΚΑΤΑΣΚΕΥΗ ΕΥΡΕΤΗΡΙΟΥ (BUILD)
Για MNIST:
python3 src/main.py
-d dataset/train-images-idx3-ubyte
-i mnist_index
-type mnist

Για SIFT:
python3 src/main.py
-d dataset/sift_learn.fvecs
-i sift_index
-type sift

Αρχεία που δημιουργούνται:
mnist_index_model.pth # Νευρονικό δίκτυο
mnist_index_index.pkl # Δεδομένα ευρετηρίου

ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ - ΑΝΑΖΗΤΗΣΗ (SEARCH)
Βεβαιωθείτε ότι έχετε δημιουργήσει ευρετήριο (βλ. Build)

Για MNIST:
python3 src/nlsh_search.py
    -d dataset/train-images.idx3-ubyte
    -q dataset/t10k-images.idx3-ubyte
    -i mnist_index
    -o results_mnist.txt
    -type mnist
    -N 10
    -T 5
    -R 2000
    -range true

Για SIFT:
python3 src/nlsh_search.py
    -d dataset/sift_learn.fvecs
    -q dataset/sift_query.fvecs
    -i sift_index
    -o results_sift.txt
    -type sift
    -N 10
    -T 5
    -R 2800
    -range true
