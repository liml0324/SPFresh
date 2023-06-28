#include "inc/Core/Common.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/VectorSetReader.h"

#include "inc/SPFresh/SPFresh.h"
#include "inc/SSDServing/SSDIndex.h"

#include <future>

#include <iomanip>
#include <iostream>
#include <fstream>

using namespace SPTAG;

namespace SPTAG {
	namespace DSPANN {
        template <typename ValueType>
        int DSPANNSearch(SPANN::Index<ValueType>* p_index) {
            SPANN::Options& p_opts = *(p_index->GetOptions());

            std::string truthFile = p_opts.m_truthPath;
            int K = p_opts.m_resultNum;
            int truthK = (p_opts.m_truthResultNum <= 0) ? K : p_opts.m_truthResultNum;

            LOG(Helper::LogLevel::LL_Info, "Loading SPANN Indices\n");

            auto querySet = SSDServing::SPFresh::LoadQuerySet(p_opts);
            int numQueries = querySet->Count();
            std::vector<QueryResult> results(numQueries, QueryResult(NULL, p_opts.m_searchInternalResultNum, false));
            std::vector<std::set<int>> visited;
            visited.resize(numQueries);

            for (int i = 0; i < p_opts.m_dspannIndexFileNum; i++) {
                std::string storePath = p_opts.m_dspannIndexFolderPrefix + "_" + std::to_string(i);
                std::shared_ptr<VectorIndex> index;
                LOG(Helper::LogLevel::LL_Info, "Loading %d SPANN Indices\n", i);
                if (index->LoadIndex(storePath, index) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to load index.\n");
                    return 1;
                }
                LOG(Helper::LogLevel::LL_Info, "Searching %d SPANN Indices\n", i);
                std::vector<QueryResult> tempResults(numQueries, QueryResult(NULL, p_opts.m_searchInternalResultNum, false));
                for (int j = 0; j < numQueries; ++j)
                {
                    tempResults[j].SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(j)));
                    tempResults[j].Reset();
                }
                std::vector<SPANN::SearchStats> stats(numQueries);
                SSDServing::SPFresh::SearchSequential((SPANN::Index<ValueType>*)index.get(), p_opts.m_searchThreadNum, tempResults, stats, p_opts.m_queryCountLimit, p_opts.m_searchInternalResultNum);
                std::string filename = p_opts.m_dspannIndexLabelPrefix + std::to_string(i);
                SPTAG::COMMON::Dataset<int> mappingData;
                LOG(Helper::LogLevel::LL_Info, "Load From %s\n", filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Info, "Initialize Mapping Error: %d\n", i);
                    exit(1);
                }
                mappingData.Load(ptr, p_opts.m_datasetRowsInBlock, p_opts.m_datasetCapacity);
                for (int index = 0; index < numQueries; index++) {
                    COMMON::QueryResultSet<ValueType>* p_queryResults = (COMMON::QueryResultSet<ValueType>*) & (tempResults[index]);
                    COMMON::QueryResultSet<ValueType>* p_queryResultsFinal = (COMMON::QueryResultSet<ValueType>*) & (results[index]);
                    for (int j = 0; j < p_queryResults->GetResultNum(); ++j) {
                        auto res = p_queryResults->GetResult(j);
                        if (res->VID == -1) break;
                        if (visited[index].find(*mappingData[res->VID]) != visited[index].end()) continue;
                        visited[index].insert(*mappingData[res->VID]);
                        p_queryResultsFinal->AddPoint(*mappingData[res->VID], res->Dist);
                    }
                }
            }
            for (int index = 0; index < numQueries; index++) {
                COMMON::QueryResultSet<ValueType>* p_queryResultsFinal = (COMMON::QueryResultSet<ValueType>*) & (results[index]);
                p_queryResultsFinal->SortResult();
            }
            std::shared_ptr<VectorSet> vectorSet;

            if (!p_opts.m_vectorPath.empty() && fileexists(p_opts.m_vectorPath.c_str())) {
                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_vectorPath))
                {
                    vectorSet = vectorReader->GetVectorSet();
                    LOG(Helper::LogLevel::LL_Info, "\nLoad VectorSet(%d,%d).\n", vectorSet->Count(), vectorSet->Dimension());
                }
            }

            float recall = 0, MRR = 0;
            std::vector<std::set<SizeType>> truth;
            if (!truthFile.empty())
            {
                LOG(Helper::LogLevel::LL_Info, "Start loading TruthFile...\n");

                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::in | std::ios::binary)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthFile.c_str());
                    exit(1);
                }
                int originalK = truthK;
                COMMON::TruthSet::LoadTruth(ptr, truth, numQueries, originalK, truthK, p_opts.m_truthType);
                char tmp[4];
                if (ptr->ReadBinary(4, tmp) == 4) {
                    LOG(Helper::LogLevel::LL_Error, "Truth number is larger than query number(%d)!\n", numQueries);
                }

                recall = COMMON::TruthSet::CalculateRecall<ValueType>(p_index, results, truth, K, truthK, querySet, vectorSet, numQueries, nullptr, false, &MRR);
                LOG(Helper::LogLevel::LL_Info, "Recall%d@%d: %f MRR@%d: %f\n", truthK, K, recall, K, MRR);
            }

            return 0;
        }


        template <typename T>
        static float CalculateRecall(VectorIndex* index, std::vector<QueryResult>& results, const std::vector<std::set<SizeType>>& truth, int K, int truthK, std::shared_ptr<SPTAG::VectorSet> querySet, std::shared_ptr<SPTAG::VectorSet> vectorSet, SizeType NumQuerys, std::vector<float>& thisrecall, std::vector<std::vector<int>>& hitTruth, std::ofstream* log = nullptr, bool debug = false, float* MRR = nullptr)
        {
            float meanrecall = 0, minrecall = MaxDist, maxrecall = 0, stdrecall = 0, meanmrr = 0;
            std::unique_ptr<bool[]> visited(new bool[K]);
            for (SizeType i = 0; i < NumQuerys; i++)
            {
                int minpos = K;
                memset(visited.get(), 0, K * sizeof(bool));
                std::vector<int> hit;
                for (SizeType id : truth[i])
                {
                    for (int j = 0; j < K; j++)
                    {
                        if (visited[j] || results[i].GetResult(j)->VID < 0) continue;

                        if (results[i].GetResult(j)->VID == id)
                        {
                            thisrecall[i] += 1;
                            visited[j] = true;
                            hit.push_back(id);
                            if (j < minpos) minpos = j;
                            break;
                        }
                        else if (vectorSet != nullptr) {
                            float dist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(results[i].GetResult(j)->VID), vectorSet->Dimension(), index->GetDistCalcMethod());
                            float truthDist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(id), vectorSet->Dimension(), index->GetDistCalcMethod());
                            if (index->GetDistCalcMethod() == SPTAG::DistCalcMethod::Cosine && fabs(dist - truthDist) < Epsilon) {
                                thisrecall[i] += 1;
                                visited[j] = true;
                                hit.push_back(id);
                                break;
                            }
                            else if (index->GetDistCalcMethod() == SPTAG::DistCalcMethod::L2 && fabs(dist - truthDist) < Epsilon * (dist + Epsilon)) {
                                thisrecall[i] += 1;
                                visited[j] = true;
                                hit.push_back(id);
                                break;
                            }
                        }
                    }
                }
                // thisrecall[i] /= truth[i].size();
                hitTruth.push_back(hit);
                meanrecall += thisrecall[i];
                if (thisrecall[i] < minrecall) minrecall = thisrecall[i];
                if (thisrecall[i] > maxrecall) maxrecall = thisrecall[i];
                if (minpos < K) meanmrr += 1.0f / (minpos + 1);

                if (debug) {
                    std::string ll("recall:" + std::to_string(thisrecall[i]) + "\ngroundtruth:");
                    std::vector<NodeDistPair> truthvec;
                    for (SizeType id : truth[i]) {
                        float truthDist = 0.0;
                        if (vectorSet != nullptr) {
                            truthDist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(id), querySet->Dimension(), index->GetDistCalcMethod());
                        }
                        truthvec.emplace_back(id, truthDist);
                    }
                    std::sort(truthvec.begin(), truthvec.end());
                    for (int j = 0; j < truthvec.size(); j++)
                        ll += std::to_string(truthvec[j].node) + "@" + std::to_string(truthvec[j].distance) + ",";
                    LOG(Helper::LogLevel::LL_Info, "%s\n", ll.c_str());
                    ll = "ann:";
                    for (int j = 0; j < K; j++)
                        ll += std::to_string(results[i].GetResult(j)->VID) + "@" + std::to_string(results[i].GetResult(j)->Dist) + ",";
                    LOG(Helper::LogLevel::LL_Info, "%s\n", ll.c_str());
                }
            }
            meanrecall /= NumQuerys;
            for (SizeType i = 0; i < NumQuerys; i++)
            {
                stdrecall += (thisrecall[i] - meanrecall) * (thisrecall[i] - meanrecall);
            }
            stdrecall = std::sqrt(stdrecall / NumQuerys);
            if (log) (*log) << meanrecall << " " << stdrecall << " " << minrecall << " " << maxrecall << std::endl;
            if (MRR) *MRR = meanmrr / NumQuerys;
            return meanrecall;
        }

        template <typename ValueType>
        int compareIndex(SPANN::Index<ValueType>* p_index_normal, SPANN::Index<ValueType>* p_index_super) {
            SPANN::Options& p_opts = *(p_index_super->GetOptions());
            std::string truthFile = p_opts.m_truthPath;
            int numThreads = p_opts.m_searchThreadNum;
            int internalResultNum = p_opts.m_searchInternalResultNum;
            int K = p_opts.m_resultNum;
            int truthK = (p_opts.m_truthResultNum <= 0) ? K : p_opts.m_truthResultNum;

            LOG(Helper::LogLevel::LL_Info, "Start loading QuerySet...\n");
            std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_queryType, p_opts.m_queryDelimiter));
            auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
            if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_queryPath))
            {
                LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                exit(1);
            }
            auto querySet = queryReader->GetVectorSet();
            int numQueries = querySet->Count();

            std::vector<QueryResult> results_normal(numQueries, QueryResult(NULL, max(K, internalResultNum), false));
            std::vector<SPANN::SearchStats> stats(numQueries);
            for (int i = 0; i < numQueries; ++i)
            {
                (*((COMMON::QueryResultSet<ValueType>*)&results_normal[i])).SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(i)), p_index_normal->m_pQuantizer);
                results_normal[i].Reset();
            }

            // p_index_super->ScanGraph();

            LOG(Helper::LogLevel::LL_Info, "Start Normal Index ANN Search...\n");

            SPTAG::SSDServing::SSDIndex::SearchSequential(p_index_normal, numThreads, results_normal, stats, p_opts.m_queryCountLimit, internalResultNum);

            LOG(Helper::LogLevel::LL_Info, "\nFinish Normal Index ANN Search...\n");

            
            std::vector<QueryResult> results_super(numQueries, QueryResult(NULL, max(K, internalResultNum), false));
            for (int i = 0; i < numQueries; ++i)
            {
                (*((COMMON::QueryResultSet<ValueType>*)&results_super[i])).SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(i)), p_index_normal->m_pQuantizer);
                results_super[i].Reset();
            }

            LOG(Helper::LogLevel::LL_Info, "Start Super Index ANN Search...\n");

            SPTAG::SSDServing::SSDIndex::SearchSequential(p_index_super, numThreads, results_super, stats, p_opts.m_queryCountLimit, internalResultNum);

            LOG(Helper::LogLevel::LL_Info, "\nFinish Super Index ANN Search...\n");
            int redundancy_normal = 0;
            int redundancy_super = 0;
            for (int i = 0; i < numQueries; ++i) {
                COMMON::QueryResultSet<ValueType>* result = (COMMON::QueryResultSet<ValueType>*) &results_normal[i];
                int lastVID;
                for (int j = 0; j < result->GetResultNum(); ++j) {
                    auto res = result->GetResult(j);
                    if (j!=0 && lastVID == res->VID) redundancy_normal++;
                    lastVID = res->VID;
                }
                result = (COMMON::QueryResultSet<ValueType>*) &results_super[i];
                for (int j = 0; j < result->GetResultNum(); ++j) {
                    auto res = result->GetResult(j);
                    if (j!=0 && lastVID == res->VID) redundancy_super++;
                    lastVID = res->VID;
                }
            }
            LOG(Helper::LogLevel::LL_Info, "Normal Redundancy: %d, Super Redundancy: %d\n", redundancy_normal, redundancy_super);
            return 0;
        }
    }
}
