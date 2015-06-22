#-------------------------------------------------
#
# Project created by QtCreator 2015-06-20T22:47:55
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = view_db
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

QMAKE_MAC_SDK = macosx10.10
QMAKE_MACOSX_DEPLOYMENT_TARGET= 10.10
QMAKE_CXXFLAGS += -stdlib=libc++
LIBS += -L/usr/local/lib -L/opt/local/lib -lcaffe -lboost_filesystem-mt -lboost_system-mt -lglog -lgflags -lprotobuf -lleveldb
INCLUDEPATH += /usr/local/include /opt/local/include /opt/local/include/eigen3 \
    /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/
QMAKE_CXXFLAGS += -Wno-unused-variable -std=c++11 -march=native #-O3
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

SOURCES += \
    view_db.cpp
