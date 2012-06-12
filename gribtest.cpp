#include "NFmiGrib.h"
#include <iostream>
#include <stdexcept>

using namespace std;

int main(int argc, char **argv) {

  if (argc != 2)
    throw runtime_error("usage: gribtest <gribfile>");

  NFmiGrib reader ;

  if (!reader.Open(static_cast<string> (argv[1])))
    throw runtime_error ("unable to Open()");

  if (!reader.NextMessage())
    throw runtime_error ("unable to NextMessage()");
    
  if (!reader.Read())
    throw runtime_error ("unable to Read()");
    
  cout << "Message " << reader.CurrentMessageIndex() << "/" << reader.MessageCount() << endl;
  
  cout << "Data datetime " << reader.DataDate() << " " << reader.DataTime() <<
  endl;
  
  cout << "Data Length " << reader.DataLength() << endl;
  
  cout << "Parameter " << endl;
  cout << " .. discipline " << reader.ParameterDiscipline() << endl;
  cout << " .. category " << reader.ParameterCategory() << endl;
  cout << " .. number " << reader.ParameterNumber() << endl;
  cout << "Grid Type " << reader.NormalizedGridType() << endl;
  cout << "X Resolution " << reader.XResolution() << endl;
  cout << "Y Resolution " << reader.YResolution() << endl;
  return 0;
} 
