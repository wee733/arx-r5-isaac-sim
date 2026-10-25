#include <arx_vla_control/core.hpp>
#include <iostream>
// JSON-lines, pure FK; no InterfacesThread and no CAN access.
int main(){vla::Kinematics kin;std::string line;
  while(std::getline(std::cin,line)) {
    try {auto q=vla::numbers<6>(vla::Json::parse(line));
      std::cout<<vla::Json({{"q",q},{"eef",vla::vector(kin.forward(q))}}).dump()<<std::endl;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
  }
}
