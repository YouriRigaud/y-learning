"""Entry for all the tests."""

#Author: Youri Rigaud
#License: MIT License

from tests.knn_tests import compare_knn_classifier, compare_knn_regressor

def main():
    """
    Main function for testing.
    """
    assert compare_knn_classifier(), "KNN classifier does not perform as well!"
    assert compare_knn_regressor(), "KNN regressor does not perform as well!"

if __name__ == "__main__":
    main()