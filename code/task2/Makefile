# Authors: Martin Bernardi & David Savary
#Feel free to buy us a cup of cofee by donating to: +34648984918 via Revolut :)

#INSTRUCTIONS:
# make: compile code
# make run: compile and run
# make clean: remove executable and binaries

# Directories
SRCDIR   = src
OBJDIR   = obj
TARGET   = ./Lab3.0AVSA2020

LINKER   = g++
CC       = g++
CFLAGS 	 = -g

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.hpp)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
rm       = rm -f

#Libraries. YOu can add extra libraries if needed
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_objdetect -lopencv_imgcodecs -lopencv_video
PATH_INCLUDES = /opt/installation/OpenCV-3.4.4/include
PATH_LIB = /opt/installation/OpenCV-3.4.4/lib

$(TARGET): $(OBJECTS)
	@$(LINKER) $(OBJECTS) -L$(PATH_LIB) $(LIBS) -o $@
	@echo "Linking complete"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	@$(CC) $(CFLAGS) -c $< -I$(PATH_INCLUDES) -o $@
	@echo "Compiled "$<""

.PHONY: clean
clean:
	@$(rm) -r $(OBJDIR)
	@$(rm) $(TARGET)
	@echo "Cleanup complete"

run: $(TARGET)
	$(TARGET)
