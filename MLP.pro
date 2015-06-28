TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    simple_MLP_baseon_function.cpp
QMAKE_CXXFLAGS+=-fopenmp
LIBS+= -lgomp -lpthread

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    read_data.h

