// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_INDEX_H_
#define _SPTAG_SPANN_INDEX_H_

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"

#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/SIMDUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/BKTree.h"
#include "inc/Core/Common/WorkSpacePool.h"

#include "inc/Core/Common/Labelset.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/Common/IQuantizer.h"

#include "IExtraSearcher.h"
#include "ExtraStaticSearcher.h"
#include "ExtraDynamicSearcher.h"
#include "Options.h"

#include <functional>
#include <shared_mutex>

namespace SPTAG
{

    namespace Helper
    {
        class IniReader;
    }

    namespace SPANN
    {
        template<typename T>
        class Index : public VectorIndex
        {
        private:
            std::shared_ptr<VectorIndex> m_index;
            std::shared_ptr<std::uint64_t> m_vectorTranslateMap;
            std::unordered_map<std::string, std::string> m_headParameters;

            std::shared_ptr<IExtraSearcher> m_extraSearcher;

            Options m_options;

            std::function<float(const T*, const T*, DimensionType)> m_fComputeDistance;
            int m_iBaseSquare;

            std::mutex m_dataAddLock;
            COMMON::VersionLabel m_versionMap;

        public:
            static thread_local std::shared_ptr<ExtraWorkSpace> m_workspace;

        public:
            Index()
            {
                m_fComputeDistance = std::function<float(const T*, const T*, DimensionType)>(COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod));
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
            }

            ~Index() {}

            inline std::shared_ptr<VectorIndex> GetMemoryIndex() { return m_index; }
            inline std::shared_ptr<IExtraSearcher> GetDiskIndex() { return m_extraSearcher; }
            inline Options* GetOptions() { return &m_options; }

            inline SizeType GetNumSamples() const { return m_versionMap.Count(); }
            inline DimensionType GetFeatureDim() const { return m_pQuantizer ? m_pQuantizer->ReconstructDim() : m_index->GetFeatureDim(); }
            inline SizeType GetValueSize() const { return m_options.m_dim * sizeof(T); }

            inline int GetCurrMaxCheck() const { return m_options.m_maxCheck; }
            inline int GetNumThreads() const { return m_options.m_iSSDNumberOfThreads; }
            inline DistCalcMethod GetDistCalcMethod() const { return m_options.m_distCalcMethod; }
            inline IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::SPANN; }
            inline VectorValueType GetVectorValueType() const { return GetEnumValueType<T>(); }
            
            void SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer);

            inline float AccurateDistance(const void* pX, const void* pY) const { 
                if (m_options.m_distCalcMethod == DistCalcMethod::L2) return m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim);

                float xy = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim);
                float xx = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pX, m_options.m_dim);
                float yy = m_iBaseSquare - m_fComputeDistance((const T*)pY, (const T*)pY, m_options.m_dim);
                return 1.0f - xy / (sqrt(xx) * sqrt(yy));
            }
            inline float ComputeDistance(const void* pX, const void* pY) const { return m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim); }
            inline bool ContainSample(const SizeType idx) const { return idx < m_options.m_vectorSize; }

            std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const
            {
                std::shared_ptr<std::vector<std::uint64_t>> buffersize(new std::vector<std::uint64_t>);
                auto headIndexBufferSize = m_index->BufferSize();
                buffersize->insert(buffersize->end(), headIndexBufferSize->begin(), headIndexBufferSize->end());
                buffersize->push_back(sizeof(long long) * m_index->GetNumSamples());
                return std::move(buffersize);
            }

            std::shared_ptr<std::vector<std::string>> GetIndexFiles() const
            {
                std::shared_ptr<std::vector<std::string>> files(new std::vector<std::string>);
                auto headfiles = m_index->GetIndexFiles();
                for (auto file : *headfiles) {
                    files->push_back(m_options.m_headIndexFolder + FolderSep + file);
                }
                if (m_options.m_excludehead) files->push_back(m_options.m_headIDFile);
                return std::move(files);
            }

            ErrorCode SaveConfig(std::shared_ptr<Helper::DiskIO> p_configout);
            ErrorCode SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams);

            ErrorCode LoadConfig(Helper::IniReader& p_reader);
            ErrorCode LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams);
            ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs);

            ErrorCode BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized = false, bool p_shareOwnership = false);
            ErrorCode BuildIndex(bool p_normalized = false);
            ErrorCode SearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const;
            ErrorCode SearchDiskIndex(QueryResult& p_query, SearchStats* p_stats = nullptr) const;
            ErrorCode DebugSearchDiskIndex(QueryResult& p_query, int p_subInternalResultNum, int p_internalResultNum,
                SearchStats* p_stats = nullptr, std::set<int>* truth = nullptr, std::map<int, std::set<int>>* found = nullptr) const;
            ErrorCode UpdateIndex();

            ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr);
            std::string GetParameter(const char* p_param, const char* p_section = nullptr) const;

            inline const void* GetSample(const SizeType idx) const { return nullptr; }
            inline SizeType GetNumDeleted() const { return m_versionMap.GetDeleteCount(); }
            inline bool NeedRefine() const { return false; }
            ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const { return ErrorCode::Undefined; }
            ErrorCode SearchTree(QueryResult& p_query) const { return ErrorCode::Undefined; }
            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false);
            ErrorCode DeleteIndex(const SizeType& p_id);

            ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum);
            ErrorCode RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams, IAbortOperation* p_abort) { return ErrorCode::Undefined; }
            ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex) { return ErrorCode::Undefined; }
            
        private:
            bool CheckHeadIndexType();
            void SelectHeadAdjustOptions(int p_vectorCount);
            int SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID, const Options& p_opts, std::vector<int>& p_selected);
            void SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount, std::vector<int>& p_selected);

            template <typename InternalDataType>
            bool SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader);

            ErrorCode BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader);

        public:
            bool AllFinished() { if (m_options.m_useKV || m_options.m_useSPDK) return m_extraSearcher->AllFinished(); return true; }

            void GetDBStat() { 
                if (m_options.m_useKV || m_options.m_useSPDK) m_extraSearcher->GetDBStats(); 
                LOG(Helper::LogLevel::LL_Info, "Current Vector Num: %d, Deleted: %d .\n", GetNumSamples(), GetNumDeleted());
            }

            void GetIndexStat(int finishedInsert, bool cost, bool reset) { if (m_options.m_useKV || m_options.m_useSPDK) m_extraSearcher->GetIndexStats(finishedInsert, cost, reset); }
            
            void ForceCompaction() { if (m_options.m_useKV) m_extraSearcher->ForceCompaction(); }

            void StopMerge() { m_options.m_inPlace = true; }

            void OpenMerge() { m_options.m_inPlace = false; }

            void ForceGC() { m_extraSearcher->ForceGC(m_index.get()); }

            bool Initialize() { return m_extraSearcher->Initialize(); }

            bool ExitBlockController() { return m_extraSearcher->ExitBlockController(); }

            ErrorCode AddIndexSPFresh(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, SizeType* VID) {
                if ((!m_options.m_useKV &&!m_options.m_useSPDK) || m_extraSearcher == nullptr) {
                    LOG(Helper::LogLevel::LL_Error, "Only Support KV Extra Update\n");
                    return ErrorCode::Fail;
                }

                if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;
                if (p_dimension != GetFeatureDim()) return ErrorCode::DimensionSizeMismatch;

                SizeType begin, end;
                {
                    std::lock_guard<std::mutex> lock(m_dataAddLock);

                    begin = m_versionMap.GetVectorNum();
                    end = begin + p_vectorNum;

                    if (begin == 0) { return ErrorCode::EmptyIndex; }

                    if (m_versionMap.AddBatch(p_vectorNum) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: VID: %d, Map Size:%d\n", begin, m_versionMap.BufferSize());
                        exit(1);
                    }
                }
                for (int i = 0; i < p_vectorNum; i++) VID[i] = begin + i;

                std::shared_ptr<VectorSet> vectorSet;
                if (m_options.m_distCalcMethod == DistCalcMethod::Cosine) {
                    ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
                    memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
                    vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
                    int base = COMMON::Utils::GetBase<T>();
                    for (SizeType i = 0; i < p_vectorNum; i++) {
                        COMMON::Utils::Normalize((T*)(vectorSet->GetVector(i)), p_dimension, base);
                    }
                }
                else {
                    vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                        GetEnumValueType<T>(), p_dimension, p_vectorNum));
                }

                return m_extraSearcher->AddIndex(vectorSet, m_index, begin);
            }

            ErrorCode MergeMultiIndex() {
                if(!m_options.m_dspann) {
                    LOG(Helper::LogLevel::LL_Error, "Not Distributed SPANN\n");
                    exit(1);
                }
                LOG(Helper::LogLevel::LL_Info, "Loading the first SPANN Index\n");
                LoadIndex(m_options.m_dspannIndexFolderPrefix + "_0" + FolderSep + m_options.m_headIndexFolder, m_index);
                if (m_options.m_useKV)
                {
                    m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options.m_KVPath.c_str(), m_options.m_dim, m_options.m_postingPageLimit * PageSize / (sizeof(T)*m_options.m_dim + sizeof(int) + sizeof(uint8_t)), m_options.m_useDirectIO, m_options.m_latencyLimit, m_options.m_mergeThreshold));
                } else {
                    LOG(Helper::LogLevel::LL_Error, "Distributed SPANN currently only support RocksDB\n");
                    exit(1);
                }
                std::shared_ptr<IExtraSearcher> storeExtraSearcher;
                storeExtraSearcher.reset(new ExtraStaticSearcher<T>());
                m_options.m_indexDirectory = m_options.m_dspannIndexFolderPrefix + "_0";
                if (!storeExtraSearcher->LoadIndex(m_options, m_versionMap)) {
                    LOG(Helper::LogLevel::LL_Info, "Initialize Error\n");
                    exit(1);
                }
                m_extraSearcher->LoadIndex(m_options, m_versionMap);
                LOG(Helper::LogLevel::LL_Info, "Initialize version map & PostingRecord\n");
                m_versionMap.Initialize(m_options.m_vectorSize, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
                m_extraSearcher->InitPostingRecord(m_index);
                std::string filenameFirst = m_options.m_dspannIndexLabelPrefix + "0";
                SPTAG::COMMON::Dataset<int> mappingDataFirst;
                LOG(Helper::LogLevel::LL_Info, "Load From %s\n", filenameFirst.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filenameFirst.c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Info, "Initialize Mapping Error: 0\n");
                    exit(1);
                }
                mappingDataFirst.Load(ptr, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
                LOG(Helper::LogLevel::LL_Info, "Writing the first index postings\n");
                int m_vectorInfoSize = sizeof(T) * m_options.m_dim + sizeof(int) + sizeof(uint8_t);

                std::vector<int> newHeadMapping;
                int length = m_index->GetNumSamples();
                newHeadMapping.resize(length);

                m_vectorTranslateMap.reset(new std::uint64_t[m_index->GetNumSamples()], std::default_delete<std::uint64_t[]>());
                std::shared_ptr<Helper::DiskIO> mptr = SPTAG::f_createIO();
                if (mptr == nullptr || !mptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                    exit(1);
                }
                IOBINARY(mptr, ReadBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), (char*)(m_vectorTranslateMap.get()));

                #pragma omp parallel for num_threads(m_options.m_iSSDNumberOfThreads) schedule(dynamic,128)
                for (int index = 0; index < m_index->GetNumSamples(); index++) {
                    newHeadMapping[index]= *mappingDataFirst[(m_vectorTranslateMap.get())[index]];
                    std::string tempPosting;
                    storeExtraSearcher->GetWritePosting(index, tempPosting);

                    int vectorNum = (int)(tempPosting.size() / (m_vectorInfoSize - sizeof(uint8_t)));

                    auto* postingP = reinterpret_cast<char*>(&tempPosting.front());
                    std::string newPosting(m_vectorInfoSize * vectorNum , '\0');
                    char* ptr = (char*)(newPosting.c_str());
                    for (int j = 0; j < vectorNum; ++j, ptr += m_vectorInfoSize) {
                        char* vectorInfo = postingP + j * (m_vectorInfoSize - sizeof(uint8_t));
                        int VID = *mappingDataFirst[*(reinterpret_cast<int*>(vectorInfo))];
                        uint8_t version = m_versionMap.GetVersion(VID);
                        memcpy(ptr, &VID, sizeof(int));
                        memcpy(ptr + sizeof(int), &version, sizeof(uint8_t));
                        memcpy(ptr + sizeof(int) + sizeof(uint8_t), vectorInfo + sizeof(int), m_vectorInfoSize - sizeof(uint8_t) - sizeof(int));
                    }

                    m_extraSearcher->GetWritePosting(index, newPosting, true);
                }
                for (int i = 1; i < m_options.m_dspannIndexFileNum; i++) {
                    LOG(Helper::LogLevel::LL_Info, "Writing the %d index postings\n", i);
                    std::string filename = m_options.m_dspannIndexLabelPrefix + std::to_string(i);
                    SPTAG::COMMON::Dataset<int> mappingData;
                    LOG(Helper::LogLevel::LL_Info, "Load From %s\n", filename.c_str());
                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) {
                        LOG(Helper::LogLevel::LL_Info, "Initialize Mapping Error: %d\n", i);
                        exit(1);
                    }
                    mappingData.Load(ptr, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
                    storeExtraSearcher.reset(new ExtraStaticSearcher<T>());
                    m_options.m_indexDirectory = m_options.m_dspannIndexFolderPrefix + "_" + std::to_string(i);
                    if (!storeExtraSearcher->LoadIndex(m_options, m_versionMap)) {
                        LOG(Helper::LogLevel::LL_Info, "Initialize Error: %d\n", i);
                        exit(1);
                    }
                    std::shared_ptr<VectorIndex> m_mergedIndex;
                    LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_mergedIndex);

                    length += m_mergedIndex->GetNumSamples();
                    newHeadMapping.resize(length);

                    m_vectorTranslateMap.reset(new std::uint64_t[m_mergedIndex->GetNumSamples()], std::default_delete<std::uint64_t[]>());

                    std::shared_ptr<Helper::DiskIO> mptr = SPTAG::f_createIO();
                    if (mptr == nullptr || !mptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                        exit(1);
                    }
                    IOBINARY(mptr, ReadBinary, sizeof(std::uint64_t) * m_mergedIndex->GetNumSamples(), (char*)(m_vectorTranslateMap.get()));
                    LOG(Helper::LogLevel::LL_Info, "Merging the %d index postings, newHeadMapping size: %d\n", i, newHeadMapping.size());
                    #pragma omp parallel for num_threads(m_options.m_iSSDNumberOfThreads) schedule(dynamic,128)
                    for (SizeType index = 0; index < m_mergedIndex->GetNumSamples(); index++) {
                        int begin, end = 0;
                        std::string tempPosting;
                        m_index->AddIndexId(m_mergedIndex->GetSample(index), 1, m_mergedIndex->GetFeatureDim(), begin, end);

                        newHeadMapping[begin]= *mappingData[(m_vectorTranslateMap.get())[index]];

                        m_index->AddIndexIdx(begin, end);

                        storeExtraSearcher->GetWritePosting(index, tempPosting);

                        int vectorNum = (int)(tempPosting.size() / (m_vectorInfoSize - sizeof(uint8_t)));

                        auto* postingP = reinterpret_cast<char*>(&tempPosting.front());
                        std::string newPosting(m_vectorInfoSize * vectorNum , '\0');
                        char* ptr = (char*)(newPosting.c_str());
                        for (int j = 0; j < vectorNum; ++j, ptr += m_vectorInfoSize) {
                            char* vectorInfo = postingP + j * (m_vectorInfoSize - sizeof(uint8_t));
                            int VID = *mappingData[*(reinterpret_cast<int*>(vectorInfo))];
                            uint8_t version = m_versionMap.GetVersion(VID);
                            memcpy(ptr, &VID, sizeof(int));
                            memcpy(ptr + sizeof(int), &version, sizeof(uint8_t));
                            memcpy(ptr + sizeof(int) + sizeof(uint8_t), vectorInfo + sizeof(int), m_vectorInfoSize - sizeof(uint8_t) - sizeof(int));
                        }

                        m_extraSearcher->GetWritePosting(begin, newPosting, true);
                    }
                }

                std::shared_ptr<Helper::DiskIO> outputIDs = SPTAG::f_createIO();
                if (outputIDs == nullptr ||
                    !outputIDs->Initialize((m_options.m_dspannIndexStoreFolder + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to create output file: %s\n",
                        (m_options.m_dspannIndexStoreFolder + FolderSep + m_options.m_headIDFile).c_str());
                    exit(1);
                }
                for (int i = 0; i < newHeadMapping.size(); i++)
                {
                    uint64_t vid = static_cast<uint64_t>(newHeadMapping[i]);
                    if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        exit(1);
                    }
                }

                m_index->SaveIndex(m_options.m_dspannIndexStoreFolder + FolderSep + m_options.m_headIndexFolder);
                m_versionMap.Save(m_options.m_deleteIDFile);
                ForceCompaction();
                return ErrorCode::Success;
            }
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_INDEX_H_
