// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Core/Common/TruthSet.h"

#include "inc/DSPANN/main.h"
#include "inc/DSPANN/DSPANNSearch.h"
 
using namespace SPTAG;

namespace SPTAG {
	namespace DSPANN {

		std::shared_ptr<VectorIndex> readIndex(std::map<std::string, std::map<std::string, std::string>>* config_map, const char* configurationPath) {
			Helper::IniReader iniReader;
			VectorValueType valueType;
			DistCalcMethod distCalcMethod;
			iniReader.LoadIniFile(configurationPath);

			(*config_map)[SEC_BASE] = iniReader.GetParameters(SEC_BASE);
			(*config_map)[SEC_SELECT_HEAD] = iniReader.GetParameters(SEC_SELECT_HEAD);
			(*config_map)[SEC_BUILD_HEAD] = iniReader.GetParameters(SEC_BUILD_HEAD);
			(*config_map)[SEC_BUILD_SSD_INDEX] = iniReader.GetParameters(SEC_BUILD_SSD_INDEX);

			valueType = iniReader.GetParameter(SEC_BASE, "ValueType", valueType);
			distCalcMethod = iniReader.GetParameter(SEC_BASE, "DistCalcMethod", distCalcMethod);
			bool buildSSD = iniReader.GetParameter(SEC_BUILD_SSD_INDEX, "isExecute", false);

			for (auto& KV : iniReader.GetParameters(SEC_SEARCH_SSD_INDEX)) {
				std::string param = KV.first, value = KV.second;
				if (buildSSD && Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "BuildSsdIndex")) continue;
				if (buildSSD && Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "isExecute")) continue;
				if (Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "PostingPageLimit")) param = "SearchPostingPageLimit";
				if (Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "InternalResultNum")) param = "SearchInternalResultNum";
				(*config_map)[SEC_BUILD_SSD_INDEX][param] = value;
			}

			std::shared_ptr<VectorIndex> index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, valueType);

			for (auto& sectionKV : *config_map) {
				for (auto& KV : sectionKV.second) {
					index->SetParameter(KV.first, KV.second, sectionKV.first);
				}
			}

			if (index->BuildIndex() != ErrorCode::Success) {
				LOG(Helper::LogLevel::LL_Error, "Failed to build index.\n");
				exit(1);
			}

			return index;
		}

		int BootProgram(const char* configurationPath_normal, const char* configurationPath_super) {
			std::map<std::string, std::map<std::string, std::string>> my_map_normal;
			std::map<std::string, std::map<std::string, std::string>> my_map_super;
			auto index_normal = readIndex(&my_map_normal, configurationPath_normal);
			auto index_super = readIndex(&my_map_super, configurationPath_super);

			#define DefineVectorValueType(Name, Type) \
			if (index_super->GetVectorValueType() == VectorValueType::Name) { \
				DSPANN::compareIndex((SPANN::Index<Type>*)(index_normal.get()), (SPANN::Index<Type>*)(index_super.get())); \
			} \

		#include "inc/Core/DefinitionList.h"
		#undef DefineVectorValueType

			return 0;
		} 
	}
}

// switch between exe and static library by _$(OutputType) 
#ifdef _exe

int main(int argc, char* argv[]) {
	if (argc < 3)
	{
		LOG(Helper::LogLevel::LL_Error,
			"ssdserving configFilePath\n");
		exit(-1);
	}
	auto ret = DSPANN::BootProgram(argv[1], argv[2]);
	return ret;
}

#endif
