#include "NFmiGrib.h"
#include <iostream>
#include <stdexcept>

using namespace std;

int main(int argc, char **argv) {

  if (argc != 2)
    throw runtime_error("usage: gribtest <gribfile>");

  NFmiGrib reader;

  if (!reader.Open(static_cast<string> (argv[1])))
    throw runtime_error ("unable to Open()");

  if (!reader.NextMessage())
    throw runtime_error ("unable to NextMessage()");
    
  cout << "Message " << reader.CurrentMessageIndex() << "/" << reader.MessageCount() << endl;
  
  cout << "Data datetime " << reader.Message()->DataDate() << " " << reader.Message()->DataTime() <<
  endl;
  
  cout << "Data Length " << reader.Message()->ValuesLength() << endl;
  
  cout << "Parameter " << endl;
  cout << " .. discipline " << reader.Message()->ParameterDiscipline() << endl;
  cout << " .. category " << reader.Message()->ParameterCategory() << endl;
  cout << " .. number " << reader.Message()->ParameterNumber() << endl;
  cout << "Grid Type " << reader.Message()->NormalizedGridType() << endl;
  cout << "X Resolution " << reader.Message()->XResolution() << endl;
  cout << "Y Resolution " << reader.Message()->YResolution() << endl;
  return 0;
} 
