#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <boost/filesystem.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;

string root_dir = "/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/";
string input_path = root_dir + "new-data-set1-writers";
string output_path = root_dir + "new-data-set1-imgs";

default_random_engine generator;

struct Word {
    string text;
    Size size;
    vector<Mat> images;
};

struct Writer {
    string id;
    vector<Word> words;
};

int w = 200, h = 70;
int min_w = 180, max_w = 240;
int counter = 0;

void saveImage(Writer &writer, Mat &textImg, vector<Word> &words) {
    int new_h = textImg.rows;
    int new_w = textImg.cols;
    Mat img = Mat::zeros(h, w, CV_8U);
    if (h < new_h) {
        if (w < new_w)
            textImg(Rect((new_w - w) / 2, (new_h - h) / 2, w, h)).copyTo(img);
        else
            textImg(Rect(0, (new_h - h) / 2, new_w, h)).copyTo(img(Rect((w - new_w) / 2, 0, new_w, h)));
    } else {
        if (w < new_w)
            textImg(Rect((new_w - w) / 2, 0, w, new_h)).copyTo(img(Rect(0, (h - new_h) / 2, w, new_h)));
        else
            textImg.copyTo(img(Rect((w - new_w) / 2, (h - new_h) / 2, new_w, new_h)));
    }

    string new_image_dir = output_path + "/" + writer.id;
    filesystem::path dirPath(new_image_dir);
    filesystem::create_directories(dirPath);

    stringstream ss;
    ss << counter;
    string new_image_file = new_image_dir + "/img" + ss.str() + ".png";

    imwrite(new_image_file, img);
}

void saveImages(vector<Writer> &writers, vector<int> &wordIndxs, Size &size) {
    cout << counter << " for words: [ ";
    for (int wordIndx: wordIndxs) {
        if (wordIndx != wordIndxs[0])
            cout << ", ";
        cout << writers[0].words[wordIndx].text;
    }
    cout << " ]" << endl;

    for (Writer &writer: writers) {
        //cout << size << endl;
        Mat textImg = Mat::zeros(size, CV_8U);
        int current_x = 0;
        vector<Word> words;
        for (int wordIndx: wordIndxs) {
            Word &word = writer.words[wordIndx];
            words.push_back(word);
            Mat wordImg;
            if (word.images.size() == 1)
                wordImg = word.images[0];
            else {
                uniform_int_distribution<int> dist_img(0, word.images.size() - 1);
                wordImg = word.images[dist_img(generator)];
            }
            //cout << wordIndx << ": " << wordImg.cols << " x " << wordImg.rows << endl;
            //cout << Rect(current_x, (textImg.rows - wordImg.rows) / 2, wordImg.cols, wordImg.rows) << endl;
            wordImg.copyTo(textImg(Rect(current_x, (textImg.rows - wordImg.rows) / 2, wordImg.cols, wordImg.rows)));
            current_x += wordImg.cols;
        }

        saveImage(writer, textImg, words);
    }
    counter++;
}

void createImages(vector<Writer> &writers, vector<Size> &wordSizes, vector<int> currentWordIndxs, Size currentSize) {
    int current_w = currentSize.width;
    int current_h = currentSize.height;
    for (int i = 0; i < wordSizes.size(); ++i) {
        bool already_in = false;
        for (int wordIndx : currentWordIndxs) {
            if (wordIndx == i) {
                already_in = true;
                break;
            }
        }
        if (already_in)
            continue;

        Size newSize;
        newSize.width = current_w + wordSizes[i].width;
        if (newSize.width > max_w && currentWordIndxs.size() > 0)
            continue;
        newSize.height = max(current_h, wordSizes[i].height);

        vector<int> newWordIndxs = currentWordIndxs;
        newWordIndxs.push_back(i);
        if (newSize.width > min_w)
            saveImages(writers, newWordIndxs, newSize);
        else
            createImages(writers, wordSizes, newWordIndxs, newSize);
    }
}

void readWord(filesystem::path word_dir, Word &word) {
    cout << "\treading word " << word.text << endl;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(word_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_regular_file(dir_iter->status())) {
            filesystem::path file_path = dir_iter->path();

            string fileName = file_path.string();
            if (fileName.rfind(".png") == string::npos)
                continue;

            Mat image = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
            word.images.push_back(image);
            word.size.width = image.cols;
            word.size.height = image.rows;
            cout << "read " << fileName << " " << word.size << endl;
        }
    }
}

void readWriter(filesystem::path writer_dir, Writer &writer) {
    cout << "reading writer " << writer.id << endl;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(writer_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            filesystem::path word_dir = dir_iter->path();

            Word word;
            word.text = word_dir.leaf().string();
            readWord(word_dir, word);
            writer.words.push_back(word);
        }
    }
}

vector<Writer> readWriters() {
    vector<Writer> writers;

    filesystem::path input_dir(input_path);
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(input_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            filesystem::path writer_dir = dir_iter->path();

            Writer writer;
            writer.id = writer_dir.leaf().string();
            readWriter(writer_dir, writer);
            writers.push_back(writer);
        }
    }
    cout << "read " << writers.size() << " writers" << endl;
    return writers;
}

vector<Size> gatherSizes(vector<Writer> &writers) {
    vector<Size> sizes;
    for (Word &word: writers[0].words) {
        cout << word.text << " with size " << word.size << endl;
        sizes.push_back(word.size);
    }
    return sizes;
}

int main(int argc, char *argv[])
{
    vector<Writer> writers = readWriters();
    vector<Size> sizes = gatherSizes(writers);
    Size initialSize(0, 0);
    vector<int> initialIndxs;
    createImages(writers, sizes, initialIndxs, initialSize);
}
