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

            ValueType* centers = (ValueType*)ALIGN_ALLOC(sizeof(ValueType) * p_opts.m_dspannIndexFileNum * p_opts.m_dim);

            {
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_opts.m_dspannCenters.c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read center file %s.\n", p_opts.m_dspannCenters.c_str());
                }

                SizeType r;
                DimensionType c;
                DimensionType col = p_opts.m_dim;
                SizeType row = p_opts.m_dspannIndexFileNum;
                ptr->ReadBinary(sizeof(SizeType), (char*)&r) != sizeof(SizeType);
                ptr->ReadBinary(sizeof(DimensionType), (char*)&c) != sizeof(DimensionType);

                if (r != row || c != col) {
                    LOG(Helper::LogLevel::LL_Error, "Row(%d,%d) or Col(%d,%d) cannot match.\n", r, row, c, col);
                }

                ptr->ReadBinary(sizeof(ValueType) * row * col, (char*)centers);
            }

            LOG(Helper::LogLevel::LL_Info, "Load Center Finished\n");

            // Top 3 selection

            int top = p_opts.m_dspannTopK;

            int needToTraverse[numQueries][top];

            struct ShardWithDist
            {
                int id;

                float dist;
            };

            std::vector<SPANN::SearchStats> stats_real(numQueries);

            for (int index = 0; index < numQueries; index++) {
                stats_real[index].m_headElementsCount = 0;
                stats_real[index].m_totalListElementsCount = 0;
                std::vector<ShardWithDist> shardDist(p_opts.m_dspannIndexFileNum);
                for (int j = 0; j < p_opts.m_dspannIndexFileNum; j++) {
                    float dist = COMMON::DistanceUtils::ComputeDistance((const ValueType*)querySet->GetVector(index), (const ValueType*)centers + j* p_opts.m_dim, querySet->Dimension(), p_index->GetDistCalcMethod());
                    shardDist[j].id = j;
                    shardDist[j].dist = dist;
                }

                std::sort(shardDist.begin(), shardDist.end(), [&](ShardWithDist& a, const ShardWithDist& b){
                    return a.dist == b.dist ? a.id < b.id : a.dist < b.dist;
                });

                for (int j = 0; j < top; j++) {
                    needToTraverse[index][j] = shardDist[j].id;
                }
            }

            LOG(Helper::LogLevel::LL_Info, "Caclulating Traverse\n");



            for (int i = 0; i < p_opts.m_dspannIndexFileNum; i++) {
                std::string storePath = p_opts.m_dspannIndexFolderPrefix + "_" + std::to_string(i);
                std::shared_ptr<VectorIndex> index;
                LOG(Helper::LogLevel::LL_Info, "Loading %d SPANN Indices: %s\n", i, storePath.c_str());
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
                SSDServing::SSDIndex::SearchSequential((SPANN::Index<ValueType>*)index.get(), p_opts.m_searchThreadNum, tempResults, stats, p_opts.m_queryCountLimit, p_opts.m_searchInternalResultNum);
                std::string filename = p_opts.m_dspannIndexLabelPrefix + std::to_string(i);
                SPTAG::COMMON::Dataset<int> mappingData;
                LOG(Helper::LogLevel::LL_Info, "Load From %s\n", filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Info, "Initialize Mapping Error: %d\n", i);
                    exit(1);
                }
                mappingData.Load(ptr, p_opts.m_datasetRowsInBlock, p_opts.m_datasetCapacity);
                // Top 1 selection
                // for (int index = 0; index < numQueries; index++) {
                //     float minDist = MaxDist;
                //     int minId;
                //     for (int j = 0; j < p_opts.m_dspannIndexFileNum; j++) {
                //         float dist = COMMON::DistanceUtils::ComputeDistance((const ValueType*)querySet->GetVector(index), (const ValueType*)centers + j* p_opts.m_dim, querySet->Dimension(), p_index->GetDistCalcMethod());
                //         if (minDist > dist) {
                //             minDist = dist;
                //             minId = j;
                //         }
                //     }
                //     if (minId != i) continue;
                //     COMMON::QueryResultSet<ValueType>* p_queryResults = (COMMON::QueryResultSet<ValueType>*) & (tempResults[index]);
                //     COMMON::QueryResultSet<ValueType>* p_queryResultsFinal = (COMMON::QueryResultSet<ValueType>*) & (results[index]);
                //     for (int j = 0; j < p_queryResults->GetResultNum(); ++j) {
                //         auto res = p_queryResults->GetResult(j);
                //         if (res->VID == -1) break;
                //         if (visited[index].find(*mappingData[res->VID]) != visited[index].end()) continue;
                //         visited[index].insert(*mappingData[res->VID]);
                //         p_queryResultsFinal->AddPoint(*mappingData[res->VID], res->Dist);
                //     }
                // }

                // Top selection
                for (int index = 0; index < numQueries; index++) {
                    bool found = false;
                    for (int j = 0; j < top; j++) {
                        if (needToTraverse[index][j] == i)
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found) continue;
                    stats_real[index].m_headElementsCount += stats[index].m_headElementsCount;
                    stats_real[index].m_totalListElementsCount += stats[index].m_totalListElementsCount;
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
            LOG(Helper::LogLevel::LL_Info, "\nHead Elements Count:\n");
            SSDServing::SSDIndex::PrintPercentiles<double, SPANN::SearchStats>(stats_real,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_headElementsCount;
                    },
                    "%.3lf");

            LOG(Helper::LogLevel::LL_Info, "\nEx Elements Count:\n");
            SSDServing::SSDIndex::PrintPercentiles<double, SPANN::SearchStats>(stats_real,
                [](const SPANN::SearchStats& ss) -> double
                {
                    return ss.m_totalListElementsCount;
                },
                "%.3lf");

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

            std::shared_ptr<VectorSet> vectorSet;

            if (!p_opts.m_vectorPath.empty() && fileexists(p_opts.m_vectorPath.c_str())) {
                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_vectorPath))
                {
                    vectorSet = vectorReader->GetVectorSet();
                    if (p_opts.m_distCalcMethod == DistCalcMethod::Cosine) vectorSet->Normalize(numThreads);
                    LOG(Helper::LogLevel::LL_Info, "\nLoad VectorSet(%d,%d).\n", vectorSet->Count(), vectorSet->Dimension());
                }
            }

            // int redundancy_normal = 0;
            // int redundancy_super = 0;
            // for (int i = 0; i < numQueries; ++i) {
            //     COMMON::QueryResultSet<ValueType>* result = (COMMON::QueryResultSet<ValueType>*) &results_normal[i];
            //     int lastVID;
            //     for (int j = 0; j < result->GetResultNum(); ++j) {
            //         auto res = result->GetResult(j);
            //         if (j!=0 && lastVID == res->VID) redundancy_normal++;
            //         lastVID = res->VID;
            //     }
            //     result = (COMMON::QueryResultSet<ValueType>*) &results_super[i];
            //     for (int j = 0; j < result->GetResultNum(); ++j) {
            //         auto res = result->GetResult(j);
            //         if (j!=0 && lastVID == res->VID) redundancy_super++;
            //         lastVID = res->VID;
            //     }
            // }
            // LOG(Helper::LogLevel::LL_Info, "Normal Redundancy: %d, Super Redundancy: %d\n", redundancy_normal, redundancy_super);

            std::vector<float> thisrecall_normal(numQueries);
            std::vector<float> thisrecall_super(numQueries);
            std::vector<std::vector<int>> hitTruth_normal;
            std::vector<std::vector<int>> hitTruth_super;

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

                recall = CalculateRecall<ValueType>((p_index_super->GetMemoryIndex()).get(), results_normal, truth, K, truthK, querySet, vectorSet, numQueries, thisrecall_normal, hitTruth_normal, nullptr, false, &MRR);
                LOG(Helper::LogLevel::LL_Info, "Nomral Recall%d@%d: %f MRR@%d: %f\n", truthK, K, recall, K, MRR);
                recall = CalculateRecall<ValueType>((p_index_super->GetMemoryIndex()).get(), results_super, truth, K, truthK, querySet, vectorSet, numQueries, thisrecall_super, hitTruth_super, nullptr, false, &MRR);
                LOG(Helper::LogLevel::LL_Info, "Super Recall%d@%d: %f MRR@%d: %f\n", truthK, K, recall, K, MRR);
            }

            //Load Partition Info

            std::vector<SPTAG::COMMON::Dataset<short>> mappingInfoVecs(32);

            for (int i = 0; i < 32; i++) {
                std::string filename = p_opts.m_dspannIndexLabelPrefix + std::to_string(i);
                LOG(Helper::LogLevel::LL_Info, "Load From %s\n", filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Info, "Initialize Mapping Error: %d\n", i);
                    exit(1);
                }
                mappingInfoVecs[i].Load(ptr, p_index_super->m_iDataBlockSize, p_index_super->m_iDataCapacity);
            }

            ValueType* centers = (ValueType*)ALIGN_ALLOC(sizeof(ValueType) * 5 * p_opts.m_dim);

            {
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_opts.m_dspannCenters.c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read center file %s.\n", p_opts.m_dspannCenters.c_str());
                }

                SizeType r;
                DimensionType c;
                DimensionType col = p_opts.m_dim;
                SizeType row = 5;
                ptr->ReadBinary(sizeof(SizeType), (char*)&r) != sizeof(SizeType);
                ptr->ReadBinary(sizeof(DimensionType), (char*)&c) != sizeof(DimensionType);

                if (r != row || c != col) {
                    LOG(Helper::LogLevel::LL_Error, "Row(%d,%d) or Col(%d,%d) cannot match.\n", r, row, c, col);
                }

                ptr->ReadBinary(sizeof(ValueType) * row * col, (char*)centers);
            }

            int count = 0;

            for (int i = 0; i < numQueries; i++) {
                if (thisrecall_normal[i] <= thisrecall_super[i]) continue;
                std::map<int, int> missStatus;
                for (auto id : truth[i]) {
                    int fileNum_id = id / 31250000;
                    int offset_id = id % 31250000;

                    QueryResult testTruth(NULL, max(K, internalResultNum), false);
                    (*((COMMON::QueryResultSet<ValueType>*)&testTruth)).SetTarget(reinterpret_cast<ValueType*>(vectorSet->GetVector(id)), p_index_normal->m_pQuantizer);
                    p_index_super->GetMemoryIndex()->SearchIndex(testTruth);
                    COMMON::QueryResultSet<ValueType>* p_queryResults = (COMMON::QueryResultSet<ValueType>*) & testTruth;

                    int miss = 0;

                    for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
                    {
                        auto res = p_queryResults->GetResult(i);
                        if (res->VID == -1) break;
                        auto realHeadID = p_index_super->ReturnTrueId(res->VID);
                        
                        int fileNum_head = realHeadID / 31250000;
                        int offset_head = realHeadID % 31250000;

                        for (int j = 0; j < 4; j++) {
                            if (mappingInfoVecs[fileNum_id][offset_id][j] > 4) continue;
                            bool found = false;
                            for (int k = 0; k < 4; k++) {
                                if (mappingInfoVecs[fileNum_head][offset_head][k] > 4) continue;
                                if (mappingInfoVecs[fileNum_head][offset_head][k] == mappingInfoVecs[fileNum_id][offset_id][j]) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                miss++;
                                break;
                            }
                        }
                    }
                    if (miss!=0) missStatus[id] = miss;
                }
                LOG(Helper::LogLevel::LL_Info, "Query: %d, BuildSSDIndex Recall: %f, Super Recall: %f\n", i, thisrecall_normal[i], thisrecall_super[i]);
                for (auto iter : missStatus) {
                    bool right = false;
                    for (int j = 0; j < hitTruth_normal[i].size(); j++) {
                        if (hitTruth_normal[i][j] == iter.first) right = true;
                    }
                    if (!right) continue; 
                    // LOG(Helper::LogLevel::LL_Info, "missStatus: %d, %d\n", iter.first, iter.second);
                    float minDist = MaxDist;
                    int minId;
                    float minSecondDist = MaxDist;
                    int minSecId;
                    for (int j = 0; j < 5; j++) {
                        float dist = COMMON::DistanceUtils::ComputeDistance((const ValueType*)vectorSet->GetVector(iter.first), (const ValueType*)centers + j* p_opts.m_dim, vectorSet->Dimension(), p_index_super->GetDistCalcMethod());
                        if (minDist > dist) {
                            minDist = dist;
                            minId = j;
                        }
                        else if (minSecondDist > dist) {
                            minSecondDist = dist;
                            minSecId = j;
                        }
                    }
                    // LOG(Helper::LogLevel::LL_Info, "first %f, second %f, factor : %f\n", minDist, minSecondDist, minSecondDist/minDist);
                    bool found = false;
                    for (int j = 0; j < hitTruth_super[i].size(); j++) {
                        if (hitTruth_super[i][j] == iter.first) found = true;
                    }
                    if (!found) {
                        count++;
                        LOG(Helper::LogLevel::LL_Info, "missStatus: %d, %d\n", iter.first, iter.second);
                        LOG(Helper::LogLevel::LL_Info, "first %f/%d, second %f/%d, factor : %f\n", minDist, minId, minSecondDist, minSecId, minSecondDist/minDist);
                        LOG(Helper::LogLevel::LL_Info, "Recall Loss\n");
                        int fileNum_id = iter.first / 31250000;
                        int offset_id = iter.first % 31250000;
                        for (int k = 0; k < 4; k++) {
                            LOG(Helper::LogLevel::LL_Info, "In %hd ", mappingInfoVecs[fileNum_id][offset_id][k]);
                        }
                        LOG(Helper::LogLevel::LL_Info, "\n");
                    }
                }
                LOG(Helper::LogLevel::LL_Info, "\n");
            }
            LOG(Helper::LogLevel::LL_Info, "Loss With Boundary issue: %d\n", count);

            // scan loss vectors
            // std::map<int, std::vector<int>> lossVectors;
            // for (int i = 0; i < numQueries; i++) {
            //     int lossNum = thisrecall_normal[i] - thisrecall_super[i];
            //     if (lossNum == 10) {
            //         std::vector<int> lossVector;
            //         for (int j = 0; j < hitTruth_normal[i].size(); j++) {
            //             bool found = false;
            //             for (int k = 0; k < hitTruth_super[i].size(); k++) {
            //                 if (hitTruth_normal[i][j] == hitTruth_super[i][j]) {
            //                     found = true;
            //                     break;
            //                 }
            //             }
            //             if (!found) lossVector.push_back(hitTruth_normal[i][j]);
            //         }
            //         lossVectors[i] = lossVector;
            //     }
            // }

            // // Load Centers, disabled first


            // // scan the status of loss vectors

            // int lossNearest = 0;

            // for (auto lossIter : lossVectors) {
            //     int queryId = lossIter.first;
            //     float minDist = 0;
            //     // Check the query;
            //     QueryResult testQuery(NULL, max(K, internalResultNum), false);
            //     (*((COMMON::QueryResultSet<ValueType>*)&testQuery)).SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(queryId)), p_index_normal->m_pQuantizer);
            //     p_index_super->GetMemoryIndex()->SearchIndex(testQuery);
            //     COMMON::QueryResultSet<ValueType>* p_queryResults = (COMMON::QueryResultSet<ValueType>*) & testQuery;
            //     for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
            //     {
            //         auto res = p_queryResults->GetResult(i);
            //         if (res->VID == -1) break;
            //         LOG(Helper::LogLevel::LL_Info, "Query Id: %d, Search Head: %d, Dist: %f\n", queryId, res->VID, res->Dist);
            //     }

            //     for (auto lossId : lossIter.second) {
            //         QueryResult testQueryLoss(NULL, max(K, internalResultNum), false);
            //         (*((COMMON::QueryResultSet<ValueType>*)&testQueryLoss)).SetTarget(reinterpret_cast<ValueType*>(vectorSet->GetVector(lossId)), p_index_normal->m_pQuantizer);
            //         p_index_super->GetMemoryIndex()->SearchIndex(testQueryLoss);
            //         p_queryResults = (COMMON::QueryResultSet<ValueType>*) & testQueryLoss;

            //         for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
            //         {
            //             auto res = p_queryResults->GetResult(i);
            //             if (res->VID == -1) break;
            //             float dist_head = COMMON::DistanceUtils::ComputeDistance((const ValueType*)querySet->GetVector(queryId), (const ValueType*)p_index_super->GetMemoryIndex()->GetSample(res->VID), vectorSet->Dimension(), p_index_super->GetDistCalcMethod());
            //             if (!lossHead && dist_head < p_queryResults->GetResult(0)->Dist) {
            //                 lossHead=true;
            //                 lossNearest++;
            //             }
            //             LOG(Helper::LogLevel::LL_Info, "Vector Id: %d, Search Head: %d, Dist: %f, ToQuery Dist: %f\n", lossId, res->VID, res->Dist, dist_head);
            //         }
            //     }
            // }
//             std::atomic_int lossNearest_super(0);
//             std::atomic_int lossNearest_normal(0);
// #pragma omp parallel for schedule(dynamic)
//             for (int queryId = 0; queryId < numQueries; queryId++) {
//                 QueryResult testQuery(NULL, max(K, internalResultNum), false);
//                 //Search Super
//                 (*((COMMON::QueryResultSet<ValueType>*)&testQuery)).SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(queryId)), p_index_normal->m_pQuantizer);
//                 p_index_super->GetMemoryIndex()->SearchIndex(testQuery);
//                 COMMON::QueryResultSet<ValueType>* p_queryResults = (COMMON::QueryResultSet<ValueType>*) & testQuery;

//                 float minDist = p_queryResults->GetResult(0)->Dist;

//                 for (auto lossId : truth[queryId]) {
//                     QueryResult testQueryLoss(NULL, max(K, internalResultNum), false);
//                     (*((COMMON::QueryResultSet<ValueType>*)&testQueryLoss)).SetTarget(reinterpret_cast<ValueType*>(vectorSet->GetVector(lossId)), p_index_normal->m_pQuantizer);
//                     p_index_super->GetMemoryIndex()->SearchIndex(testQueryLoss);
//                     p_queryResults = (COMMON::QueryResultSet<ValueType>*) & testQueryLoss;

//                     for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
//                     {
//                         auto res = p_queryResults->GetResult(i);
//                         if (res->VID == -1) break;
//                         float dist_head = COMMON::DistanceUtils::ComputeDistance((const ValueType*)querySet->GetVector(queryId), (const ValueType*)p_index_super->GetMemoryIndex()->GetSample(res->VID), vectorSet->Dimension(), p_index_super->GetDistCalcMethod());
//                         if (dist_head < minDist) {
//                             lossNearest_super++;
//                         }
//                         // LOG(Helper::LogLevel::LL_Info, "Vector Id: %d, Search Head: %d, Dist: %f, ToQuery Dist: %f\n", lossId, res->VID, res->Dist, dist_head);
//                     }
//                 }
                

//                 (*((COMMON::QueryResultSet<ValueType>*)&testQuery)).SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(queryId)), p_index_normal->m_pQuantizer);
//                 testQuery.Reset();
//                 p_index_normal->GetMemoryIndex()->SearchIndex(testQuery);
//                 p_queryResults = (COMMON::QueryResultSet<ValueType>*) & testQuery;

//                 minDist = p_queryResults->GetResult(0)->Dist;

//                 for (auto lossId : truth[queryId]) {
//                     QueryResult testQueryLoss(NULL, max(K, internalResultNum), false);
//                     (*((COMMON::QueryResultSet<ValueType>*)&testQueryLoss)).SetTarget(reinterpret_cast<ValueType*>(vectorSet->GetVector(lossId)), p_index_normal->m_pQuantizer);
//                     p_index_normal->GetMemoryIndex()->SearchIndex(testQueryLoss);
//                     p_queryResults = (COMMON::QueryResultSet<ValueType>*) & testQueryLoss;

//                     for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
//                     {
//                         auto res = p_queryResults->GetResult(i);
//                         if (res->VID == -1) break;
//                         float dist_head = COMMON::DistanceUtils::ComputeDistance((const ValueType*)querySet->GetVector(queryId), (const ValueType*)p_index_normal->GetMemoryIndex()->GetSample(res->VID), vectorSet->Dimension(), p_index_super->GetDistCalcMethod());
//                         if (dist_head < minDist) {
//                             lossNearest_normal++;
//                         }
//                         // LOG(Helper::LogLevel::LL_Info, "Vector Id: %d, Search Head: %d, Dist: %f, ToQuery Dist: %f\n", lossId, res->VID, res->Dist, dist_head);
//                     }
//                 }
//                 //Search Normal
//             }

//             LOG(Helper::LogLevel::LL_Info, "lossNearest Head Super : %d, lossNearest Head Normal : %d\n", lossNearest_super.load(), lossNearest_normal.load());

            return 0;
        }
    }
}
