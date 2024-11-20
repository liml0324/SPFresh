#ifndef _SPTAG_SPANN_EXTRAFILECONTROLLER_H_
#define _SPTAG_SPANN_EXTRAFILECONTROLLER_H_
#include "inc/Helper/KeyValueIO.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ThreadPool.h"
#include <cstdlib>
#include <memory>
#include <atomic>
#include <mutex>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>
namespace SPTAG::SPANN {
    typedef std::int64_t AddressType;
    class FileIO : public Helper::KeyValueIO {
        class BlockController {
        private:
            static char* filePath;
            static constexpr AddressType kSsdImplMaxNumBlocks = (300ULL << 30) >> PageSizeEx; // 300G
            static constexpr const char* kFileIoDepth = "SPFRESH_FILE_IO_DEPTH";
            static constexpr int kSsdFileIoDefaultIoDepth = 1024;

            tbb::concurrent_queue<AddressType> m_blockAddresses;
            tbb::concurrent_queue<AddressType> m_blockAddresses_reserve;
            
            pthread_t m_fileIoTid;
            volatile bool m_fileIoThreadStartFailed = false;
            volatile bool m_fileIoThreadReady = false;
            volatile bool m_fileIoThreadExiting = false;

            int m_ssdFileIoDepth = kSsdFileIoDefaultIoDepth;
            struct SubIoRequest {
                tbb::concurrent_queue<SubIoRequest *>* completed_sub_io_requests;
                void* app_buff;
                void* dma_buff;
                AddressType real_size;
                AddressType offset;
                bool is_read;
                BlockController* ctrl;
                int posting_id;
            };
            tbb::concurrent_queue<SubIoRequest *> m_submittedSubIoRequests;
            struct IoContext {
                std::vector<SubIoRequest> sub_io_requests;
                std::vector<SubIoRequest *> free_sub_io_requests;
                tbb::concurrent_queue<SubIoRequest *> completed_sub_io_requests;
                int in_flight = 0;
            };
            static thread_local struct IoContext m_currIoContext;

            static int m_ssdInflight;

            static std::unique_ptr<char[]> m_memBuffer;

            std::mutex m_initMutex;
            int m_numInitCalled = 0;

            int m_batchSize;
            static int m_ioCompleteCount;
            int m_preIOCompleteCount = 0;
            std::chrono::time_point<std::chrono::high_resolution_clock> m_preTime = std::chrono::high_resolution_clock::now();

            static void* InitializeFileIo(void* args);

            static void Start(void* args);

            static void FileIoLoop(void *arg);

            static void FileIoCallback(bool success, void *cb_arg);

            static void Stop(void* args);

        public:
            bool Initialize(int batchSize);

            bool GetBlocks(AddressType* p_data, int p_size);

            bool ReleaseBlocks(AddressType* p_data, int p_size);

            bool ReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value);

            bool IOStatistics();

            bool ShutDown();

            int RemainBlocks() {
                return m_blockAddresses.unsafe_size();
            };
        };

    private:
        std::string m_mappingPath;
        SizeType m_blockLimit;
        COMMON::Dataset<AddressType> m_pBlockMapping;
        SizeType m_bufferLimit;
        tbb::concurrent_queue<AddressType> m_buffer;

        std::shared_ptr<Helper::ThreadPool> m_compactionThreadPool;
        BlockController m_pBlockController;

        bool m_shutdownCalled;
        std::mutex m_updateMutex;
    };
}
#endif