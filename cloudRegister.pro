QT = core

CONFIG += c++17 cmdline

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
msvc: {
    QMAKE_CXXFLAGS += /utf-8
    QMAKE_CFLAGS   += /utf-8
}
SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32: LIBS += -LE:/ThirdParty/opencv-4.7.0-NoCudda/opencv/build/x64/vc16/lib/ -lopencv_world470

INCLUDEPATH += E:/ThirdParty/opencv-4.7.0-NoCudda/opencv/build/include
DEPENDPATH += E:/ThirdParty/opencv-4.7.0-NoCudda/opencv/build/include

INCLUDEPATH += E:/ThirdParty/eigen-3.3.9
