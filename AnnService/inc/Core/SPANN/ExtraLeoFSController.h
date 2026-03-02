#ifndef _SPTAG_SPANN_EXTRALEOFSCONTROLLER_H_
#define _SPTAG_SPANN_EXTRALEOFSCONTROLLER_H_
#define USE_ASYNC_IO
// #define USE_FILE_DEBUG
#include "inc/Helper/KeyValueIO.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ThreadPool.h"
#include "Options.h"
#include "leofs.h"
#include <cstdlib>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <fcntl.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>
#include <sys/syscall.h>
#include <list>
namespace SPTAG::SPANN {
    typedef std::int64_t AddressType;
    class LeoFSIO : public Helper::KeyValueIO {
        class BlockController {
        private:
            class WriteBuffer {
            private:
                std::queue<char*> m_bufQueue;
                std::unordered_map<AddressType, std::pair<char*, int>> m_buffer[2];
                std::unordered_map<AddressType, std::pair<char*, int>> *m_pCurrBuffer;
                std::unordered_map<AddressType, std::pair<char*, int>> *m_pDumpBuffer;
                std::shared_mutex m_mutex;
                std::shared_mutex m_dumpMutex;
                std::mutex m_queueMutex;
                std::vector<int> m_cids;
                std::vector<int> m_fds;
                std::condition_variable m_cv;
                bool m_isDumping;
                int m_dumpThreadNum;
                int m_pageSize;
                int m_bufferSize;
                int m_batchSize;
                std::atomic<int64_t> m_dumpedBlocks;
                std::atomic<int64_t> m_dumpedTime;
                pthread_t m_pDumpMainThread;
            public:
                WriteBuffer(int pageSize, int bufferSize, int dumpThreadNum, const std::string &leofsConfigPath, const char* filePath, int batchSize) {
                    m_pageSize = pageSize;
                    m_bufferSize = bufferSize / 2;
                    m_dumpThreadNum = dumpThreadNum;
                    m_pCurrBuffer = &m_buffer[0];
                    m_pDumpBuffer = &m_buffer[1];
                    m_isDumping = false;
                    m_batchSize = batchSize;
                    m_dumpedBlocks = 0;
                    m_dumpedTime = 0;
                    m_pDumpMainThread = -1;
                    for (int i = 0; i < 2 * m_bufferSize; i++) {
                        char* buffer = new char[m_pageSize];
                        m_bufQueue.push(buffer);
                    }
                    m_cids.resize(m_dumpThreadNum, -1);
                    m_fds.resize(m_dumpThreadNum, -1);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBuffer: filePath: %s\n", filePath);
                    for (int i = 0; i < m_dumpThreadNum; i++) {
                        auto cid = dfs_connect_config(leofsConfigPath.c_str());
                        if (cid < 0) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "WriteBuffer: connect to leofs failed\n");
                            exit(0);
                        }
                        m_cids[i] = cid;
                        auto fd = dfs_open(cid, filePath, O_WRONLY | O_CREAT, 0644);
                        if (fd < 0) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "WriteBuffer: open file failed\n");
                            exit(0);
                        }
                        m_fds[i] = fd;
                    }
                };

                ~WriteBuffer() {
                    std::unique_lock<std::shared_mutex> lock(m_mutex);
                    std::unique_lock<std::shared_mutex> dumpLock(m_dumpMutex);
                    std::unique_lock<std::mutex> queueLock(m_queueMutex);
                    // TODO: consider dump buffer here
                    if (m_isDumping) {
                        void *ret_val = nullptr;
                        pthread_join(m_pDumpMainThread, &ret_val);
                        m_isDumping = false;
                    }
                    while (!m_bufQueue.empty()) {
                        char* bufptr = m_bufQueue.front();
                        m_bufQueue.pop();
                        delete[] bufptr;
                    }
                    for (auto &it : m_buffer[0]) {
                        delete[] it.second.first;
                    }
                    for (auto &it : m_buffer[1]) {
                        delete[] it.second.first;
                    }
                    for (int i = 0; i < m_dumpThreadNum; i++) {
                        if (m_cids[i] >= 0) {
                            if (m_fds[i] >= 0) {
                                dfs_close(m_cids[i], m_fds[i]);
                            }
                            dfs_disconnect(m_cids[i]);
                        }
                    }
                }

                static void dumpThread(std::vector<std::pair<AddressType, std::pair<char*, int>>> &dumpJobs, int cid, int fd, int batchSize) {
                    int totalSize = dumpJobs.size();
                    std::vector<dfs_iocb> iocbs;
                    std::vector<dfs_iocb*> iocb_ptr;
                    iocbs.resize(batchSize);
                    iocb_ptr.resize(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        iocb_ptr[i] = &iocbs[i];
                    }
                    for (int i = 0; i < totalSize; i += batchSize) {
                        int batch = std::min(batchSize, totalSize - i);
                        for (int j = 0; j < batch; j++) {
                            auto &it = dumpJobs[i + j];
                            iocbs[j].aio_lio_opcode = 1;
                            iocbs[j].aio_fildes = fd;
                            iocbs[j].aio_buf = reinterpret_cast<void*>(it.second.first);
                            iocbs[j].aio_nbytes = it.second.second;
                            iocbs[j].aio_offset = it.first;
                        }
                        dfs_multi_pwrite(cid, fd, batch, iocb_ptr.data());
                    }
                    dfs_fsync(cid, fd);
                }

                static void* dump(void *args) {
                    auto begin = std::chrono::high_resolution_clock::now();
                    WriteBuffer *wb = static_cast<WriteBuffer*>(args);
                    std::vector<std::vector<std::pair<AddressType, std::pair<char*, int>>>> dumpJobs(wb->m_dumpThreadNum);
                    int i = 0;
                    for (auto &it : *(wb->m_pDumpBuffer)) {
                        dumpJobs[i % wb->m_dumpThreadNum].push_back(it);
                        i++;
                    }
                    std::thread dumpThreads[wb->m_dumpThreadNum];
                    for (int i = 0; i < wb->m_dumpThreadNum; i++) {
                        dumpThreads[i] = std::thread(dumpThread, std::ref(dumpJobs[i]), wb->m_cids[i], wb->m_fds[i], wb->m_batchSize);
                    }
                    for (int i = 0; i < wb->m_dumpThreadNum; i++) {
                        dumpThreads[i].join();
                    }
                    std::unique_lock<std::shared_mutex> lock(wb->m_dumpMutex);
                    std::unique_lock<std::mutex> queueLock(wb->m_queueMutex);
                    for (auto &it : *(wb->m_pDumpBuffer)) {
                        wb->m_bufQueue.push(it.second.first);
                    }
                    wb->m_pDumpBuffer->clear();
                    // wb->m_isDumping = false;
                    // wb->m_cv.notify_all();
                    auto end = std::chrono::high_resolution_clock::now();
                    wb->m_dumpedTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    return nullptr;
                }

                void dumpAsync() {
                    std::unique_lock<std::shared_mutex> lock(m_dumpMutex);
                    m_isDumping = true;
                    std::swap(m_pCurrBuffer, m_pDumpBuffer);
                    if (m_pDumpBuffer->size() == 0) {
                        // m_isDumping = false;
                        return;
                    }
                    m_dumpedBlocks += m_pDumpBuffer->size();
                    pthread_create(&m_pDumpMainThread, nullptr, dump, this);
                }

                void forceDump() {
                    std::unique_lock<std::shared_mutex> lock(m_mutex);
                    if (m_isDumping) {
                        void *ret_val = nullptr;
                        pthread_join(m_pDumpMainThread, &ret_val);
                        m_isDumping = false;
                    }
                    if (m_pDumpBuffer->size() > 0) {
                        pthread_create(&m_pDumpMainThread, nullptr, dump, this);
                        void *ret_val = nullptr;
                        pthread_join(m_pDumpMainThread, &ret_val);
                        m_pDumpMainThread = -1;
                    }
                    {
                        std::unique_lock<std::shared_mutex> lock(m_dumpMutex);
                        std::swap(m_pCurrBuffer, m_pDumpBuffer);
                    }
                    if (m_pDumpBuffer->size() > 0) {
                        pthread_create(&m_pDumpMainThread, nullptr, dump, this);
                        void *ret_val = nullptr;
                        pthread_join(m_pDumpMainThread, &ret_val);
                        m_pDumpMainThread = -1;
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBuffer: Force dump done!\n");
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBuffer: Current buffer size: %d\n", m_pCurrBuffer->size());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBuffer: Dump buffer size: %d\n", m_pDumpBuffer->size());
                }

                bool put(AddressType addr, void *value, int size) {
                    std::unique_lock<std::shared_mutex> lock(m_mutex);
                    
                    if (size > m_pageSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "WriteBuffer: Put size too large!");
                        return false;
                    }
                    if (m_pCurrBuffer->size() >= m_bufferSize) {
                        if (m_isDumping) {
                            void *ret_val = nullptr;
                            pthread_join(m_pDumpMainThread, &ret_val);
                            m_isDumping = false;
                        }
                        dumpAsync();
                    }
                    auto it = m_pCurrBuffer->find(addr);
                    if (it != m_pCurrBuffer->end()) {
                        auto ptr = it->second.first;
                        memcpy(ptr, value, size);
                        it->second.second = size;
                        return true;
                    }
                    char *buf_ptr = nullptr;
                    {
                        std::unique_lock<std::mutex> lock(m_queueMutex);
                        if (m_bufQueue.size() > 0) {
                            buf_ptr = m_bufQueue.front();
                            m_bufQueue.pop();
                        } else {
                            return false;
                        }
                    }
                    memcpy(buf_ptr, value, size);
                    m_pCurrBuffer->insert(std::make_pair(addr, std::pair<char*, int>(buf_ptr, size)));
                    return true;
                }

                bool get(AddressType addr, void *value, int size) {
                    {
                        std::shared_lock<std::shared_mutex> lock(m_mutex);
                        auto it = m_pCurrBuffer->find(addr);
                        if (it != m_pCurrBuffer->end()) {
                            if (it->second.second != size) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "WriteBuffer: Get size inconsistent!");
                            }
                            memcpy(value, it->second.first, std::min(it->second.second, size));
                            return true;
                        }
                    }
                    
                    {
                        std::shared_lock<std::shared_mutex> lock(m_dumpMutex);
                        auto it = m_pDumpBuffer->find(addr);
                        if (it != m_pDumpBuffer->end()) {
                            if (it->second.second != size) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "WriteBuffer: Get size inconsistent!");
                            }
                            memcpy(value, it->second.first, std::min(it->second.second, size));
                            return true;
                        }
                    }
                    
                    return false;
                }

                void getStats() {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBuffer: Buffer dump using time: %lld\n", m_dumpedTime.load());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBuffer: Buffer dumped block count: %lld\n", m_dumpedBlocks.load());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBuffer: Time for each block: %lf\n", (double)m_dumpedTime / m_dumpedBlocks);
                    m_dumpedTime.store(0);
                    m_dumpedBlocks.store(0);
                }
            };
            static constexpr const char* kLeoFSPath = "SPFRESH_LEOFS_IO_PATH";
            static constexpr const char* kLeoFSConfigPath = "SPFRESH_LEOFS_IO_CONFIG_PATH";
            std::string m_LeoFSConfigPath;
            static char* filePath;
            static thread_local int fd;
            static thread_local int cid;
            tbb::concurrent_queue<std::pair<int, int>> m_cidFds;

            static constexpr AddressType kSsdImplMaxNumBlocks = (300ULL << 30) >> PageSizeEx; // 300G
            static constexpr const char* kLeoFSDepth = "SPFRESH_LEOFS_IO_DEPTH";
            static constexpr int kSsdLeoFSDefaultIoDepth = 1024;
            static constexpr const char* kLeoFSThreadNum = "SPFRESH_LEOFS_IO_THREAD_NUM";
            static constexpr int kSsdLeoFSDefaultIoThreadNum = 64;
            static constexpr const char* kLeoFSAlignment = "SPFRESH_LEOFS_IO_ALIGNMENT";
            static constexpr int kSsdLeoFSDefaultAlignment = 4096;

            WriteBuffer *m_pWriteBuffer = nullptr;
            // RWThreadPool *m_prwThreadPool;
            static thread_local std::atomic_uint32_t inflight_jobs;

            tbb::concurrent_queue<AddressType> m_blockAddresses;
            tbb::concurrent_queue<AddressType> m_blockAddresses_reserve;
            
            pthread_t m_LeoFSTid;
            pthread_t m_ioStatisticsTid;
            volatile bool m_LeoFSThreadStartFailed = false;
            volatile bool m_LeoFSThreadReady = false;
            volatile bool m_LeoFSThreadExiting = false;

            int m_ssdLeoFSAlignment = kSsdLeoFSDefaultAlignment;
            int m_ssdLeoFSDepth = kSsdLeoFSDefaultIoDepth;
            int m_ssdLeoFSThreadNum = kSsdLeoFSDefaultIoThreadNum;
            struct SubIoRequest {
                dfs_iocb myiocb;
                AddressType real_size;
                AddressType offset;
                void* app_buff;
                BlockController* ctrl;
                int posting_id;
            };

            static thread_local void* aligned_buf;
            tbb::concurrent_queue<SubIoRequest *> m_submittedSubIoRequests;
            struct IoContext {
                std::vector<SubIoRequest> sub_io_requests;
                std::queue<SubIoRequest *> free_sub_io_requests;
                int in_flight = 0;
            };
            static thread_local struct IoContext m_currIoContext;
            static thread_local int debug_fd;
            static thread_local uint64_t iocp;
            static std::chrono::high_resolution_clock::time_point m_startTime;

            Options *m_pOpt;

            static thread_local int id;
            int m_maxId = 0;
            std::queue<int> m_idQueue;
            std::vector<int> read_complete_vec;
            std::vector<int> read_submit_vec;
            std::vector<int> write_complete_vec;
            std::vector<int> write_submit_vec;
            std::vector<int64_t> read_bytes_vec;
            std::vector<int64_t> write_bytes_vec;
            std::vector<int64_t> read_blocks_time_vec;

            std::vector<int64_t> multi_read_time_vec;
            std::vector<int64_t> multi_read_times;

            std::mutex m_uniqueResourceMutex;

            static int m_ssdInflight;

            static std::unique_ptr<char[]> m_memBuffer;

            std::mutex m_initMutex;
            int m_numInitCalled = 0;

            int m_batchSize;
            static std::atomic<int> m_ioCompleteCount;
            int m_preIOCompleteCount = 0;
            int64_t m_preIOBytes = 0;
            std::chrono::time_point<std::chrono::high_resolution_clock> m_preTime = std::chrono::high_resolution_clock::now();

            std::atomic<int64_t> m_batchReadTimes;
            std::atomic<int64_t> m_batchReadTimeouts;

            static void* InitializeLeoFS(void* args);

            static void* IoStatisticsThread(void* args) {
                auto ctrl = static_cast<BlockController*>(args);
                pthread_exit(NULL);
            };

            // static void Start(void* args);

            // static void LeoFSLoop(void *arg);

            // static void LeoFSCallback(bool success, void *cb_arg);

            // static void Stop(void* args);

        public:
            bool Initialize(int batchSize);

            bool GetBlocks(AddressType* p_data, int p_size);

            bool ReleaseBlocks(AddressType* p_data, int p_size);

            bool ReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool NewReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool BufferedReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocks(AddressType* p_data, ByteArray& p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<ByteArray>& p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool NewReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocksAsync(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool BufferedReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value);

            bool WriteBlocks(AddressType* p_data, int p_size, const ByteArray& p_value);

            bool NewWriteBlocks(AddressType* p_data, int p_size, const std::string& p_value);

            bool BufferedWriteBlocks(AddressType* p_data, int p_size, const std::string& p_value);

            bool IOStatistics();

            bool ShutDown();

            void SetWriteBuffer(Options *opt) {
                m_pWriteBuffer = new WriteBuffer(PageSize, opt->m_writeBufferSize, opt->m_writeBufferDumpThreadNum, m_LeoFSConfigPath, filePath, m_batchSize);
            };

            int RemainBlocks() {
                return m_blockAddresses.unsafe_size();
            };

            void SetOpt(Options *opt) {
                m_pOpt = opt;
            }

            int GetCid() {
                return cid;
            }

            ErrorCode Checkpoint(std::string prefix) {
                // TODO: Consider checkpoint to dfs
                if (m_pWriteBuffer) {
                    m_pWriteBuffer->forceDump();
                }
                std::string filename = prefix + "_blockpool";
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO: saving block pool\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Reload reserved blocks!\n");
                AddressType currBlockAddress = 0;
                for (int count = 0; count < m_blockAddresses_reserve.unsafe_size(); count++) {
                    m_blockAddresses_reserve.try_pop(currBlockAddress);
                    m_blockAddresses.push(currBlockAddress);
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Reload Finish!\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save blockpool To %s\n", filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                int blocks = RemainBlocks();
                IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&blocks);
                for (auto it = m_blockAddresses.unsafe_begin(); it != m_blockAddresses.unsafe_end(); it++) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType), (char*)&(*it));
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save Finish!\n");
                return ErrorCode::Success;
            }

            ErrorCode CheckpointDFS(std::string prefix) {
                std::string filename = prefix + "_blockpool";
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO: saving block pool to DFS\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Reload reserved blocks!\n");
                AddressType currBlockAddress = 0;
                for (int count = 0; count < m_blockAddresses_reserve.unsafe_size(); count++) {
                    m_blockAddresses_reserve.try_pop(currBlockAddress);
                    m_blockAddresses.push(currBlockAddress);
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Reload Finish!\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save blockpool To %s\n", filename.c_str());
                auto ptr = f_createDFSIO();
                if (ptr == nullptr || !ptr->Initialize(-1, m_pOpt->m_leoFSConfigPath.c_str(), filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644)) return ErrorCode::FailedCreateFile;
                int blocks = RemainBlocks();
                IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&blocks);
                for (auto it = m_blockAddresses.unsafe_begin(); it != m_blockAddresses.unsafe_end(); it++) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType), (char*)&(*it));
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save Finish!\n");
                return ErrorCode::Success;
            }

            ErrorCode Recovery(std::string prefix, int batchSize) {
                std::lock_guard<std::mutex> lock(m_initMutex);
                m_numInitCalled++;
                // TODO: Consider recovery from dfs
                
                int blocks;
                
                AddressType currBlockAddress = 0;

                if (m_pOpt->m_recoverFromLeoFS) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO Recovery: Loading block pool from DFS\n");
                    std::string filename = prefix + "_blockpool";
                    auto ptr = f_createDFSIO();
                    if (ptr == nullptr || !ptr->Initialize(-1, m_pOpt->m_leoFSConfigPath.c_str(), filename.c_str(), O_RDONLY, 0644)) {
                        return ErrorCode::FailedCreateFile;
                    }
                    IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&blocks);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO Recovery: Reading %d blocks to pool from DFS\n", blocks);
                    for (int i = 0; i < blocks; i++) {
                        IOBINARY(ptr, ReadBinary, sizeof(AddressType), (char*)&(currBlockAddress));
                        m_blockAddresses.push(currBlockAddress);
                    }
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO Recovery: Loading block pool\n");
                    std::string filename = prefix + "_blockpool";
                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) {
                        return ErrorCode::FailedCreateFile;
                    }
                    IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&blocks);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO Recovery: Reading %d blocks to pool\n", blocks);
                    for (int i = 0; i < blocks; i++) {
                        IOBINARY(ptr, ReadBinary, sizeof(AddressType), (char*)&(currBlockAddress));
                        m_blockAddresses.push(currBlockAddress);
                    }
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO Recovery: Initializing LeoFSIO\n");
                
                if (m_numInitCalled == 1) {
                    m_batchSize = batchSize;
                    m_startTime = std::chrono::high_resolution_clock::now();
                    pthread_create(&m_LeoFSTid, NULL, &InitializeLeoFS, this);
                    while(!m_LeoFSThreadReady && !m_LeoFSThreadStartFailed);
                    if (m_LeoFSThreadStartFailed) {
                        fprintf(stderr, "SPDKIO::BlockController::Initialize failed\n");
                        return ErrorCode::Fail;
                    }
                }
                const char* LeoFSConfigPath = getenv(kLeoFSConfigPath);
                if (!LeoFSConfigPath) {
                    fprintf(stderr, "LeoFSIO::BlockController::Initialize failed: LeoFSConfigPath is not set\n");
                    return ErrorCode::Fail;
                }
                cid = dfs_connect_config(LeoFSConfigPath);
                if (cid < 0) {
                    fprintf(stderr, "LeoFSIO::BlockController::Initialize failed: dfs_connect_config failed\n");
                    return ErrorCode::Fail;
                }       

                fd = dfs_open(cid, filePath, O_RDWR | O_DIRECT, 0666);
                if (fd < 0) {
                    auto err_str = dfs_errno(cid);
                    fprintf(stderr, "open failed: %d\n", err_str);
                    return ErrorCode::Fail;
                }

                aligned_buf = aligned_alloc(m_ssdLeoFSAlignment, PageSize);

                if (m_idQueue.empty()) {
                    id = m_maxId;
                    m_maxId++;
                }
                else {
                    id = m_idQueue.front();
                    m_idQueue.pop();
                }
                while(read_complete_vec.size() <= id) {
                    read_complete_vec.push_back(0);
                }
                while(read_submit_vec.size() <= id) {
                    read_submit_vec.push_back(0);
                }
                while(write_complete_vec.size() <= id) {
                    write_complete_vec.push_back(0);
                }
                while(write_submit_vec.size() <= id) {
                    write_submit_vec.push_back(0);
                }
                while(read_bytes_vec.size() <= id) {
                    read_bytes_vec.push_back(0);
                }
                while(write_bytes_vec.size() <= id) {
                    write_bytes_vec.push_back(0);
                }
                while(read_blocks_time_vec.size() <= id) {
                    read_blocks_time_vec.push_back(0);
                }
                // Create sub I/O request pool
                m_currIoContext.sub_io_requests.resize(m_ssdLeoFSDepth);
                m_currIoContext.in_flight = 0;
                for (auto &sr : m_currIoContext.sub_io_requests) {
                    sr.app_buff = nullptr;
                    auto buf_ptr = aligned_alloc(m_ssdLeoFSAlignment, PageSize);
                    if (buf_ptr == nullptr) {
                        fprintf(stderr, "LeoFSIO::BlockController::Initialize failed: aligned_alloc failed\n");
                        return ErrorCode::Fail;
                    }
                    sr.myiocb.aio_buf = buf_ptr;
                    sr.myiocb.aio_fildes = fd;
                    sr.myiocb.aio_data = reinterpret_cast<uintptr_t>(&sr);
                    sr.myiocb.aio_nbytes = PageSize;
                    sr.ctrl = this;
                    m_currIoContext.free_sub_io_requests.push(&sr);
                }
                iocp = 0;
                auto ret = dfs_io_setup(cid, m_ssdLeoFSDepth, &iocp);
                if (ret < 0) {
                    fprintf(stderr, "LeoFSIO::BlockController::Initialize io_setup failed: %s\n", strerror(errno));
                    fprintf(stderr, "m_ssdLeoFSDepth = %d, iocp = %p\n", m_ssdLeoFSDepth, &iocp);
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        };

        class MergeBuffer {
        private:
            class IndexedHeap {
            public:
                struct Node {
                    int key;   // 优先级
                    int id;    // 唯一标识
                };

                std::vector<Node> h;     // 堆数组
                std::vector<int> pos;    // pos[id] = 在堆中的下标，-1 表示不存在

                IndexedHeap(int max_id) {
                    pos.assign(max_id + 1, -1);
                }

                bool empty() const {
                    return h.empty();
                }

                int size() const {
                    return h.size();
                }

                const Node& top() const {
                    return h[0];
                }

                /* ---------- 内部工具函数 ---------- */

                void swap_node(int i, int j) {
                    std::swap(h[i], h[j]);
                    pos[h[i].id] = i;
                    pos[h[j].id] = j;
                }

                void sift_up(int i) {
                    while (i > 0) {
                        int p = (i - 1) / 2;
                        if (h[p].key >= h[i].key) break;
                        swap_node(p, i);
                        i = p;
                    }
                }

                void sift_down(int i) {
                    int n = h.size();
                    while (true) {
                        int l = i * 2 + 1;
                        int r = i * 2 + 2;
                        int largest = i;

                        if (l < n && h[l].key > h[largest].key)
                            largest = l;
                        if (r < n && h[r].key > h[largest].key)
                            largest = r;

                        if (largest == i) break;
                        swap_node(i, largest);
                        i = largest;
                    }
                }

                /* ---------- 对外接口 ---------- */

                // 插入新元素
                void push(int id, int key) {
                    if (pos[id] != -1) return;  // 已存在
                    h.push_back({key, id});
                    pos[id] = h.size() - 1;
                    sift_up(pos[id]);
                }

                // 弹出堆顶
                void pop() {
                    if (h.empty()) return;
                    int last = h.size() - 1;
                    swap_node(0, last);
                    pos[h[last].id] = -1;
                    h.pop_back();
                    if (!h.empty())
                        sift_down(0);
                }

                // 修改任意元素的 key
                void modify(int id, int new_key) {
                    int i = pos[id];
                    if (i == -1) return;

                    int old = h[i].key;
                    h[i].key = new_key;

                    if (new_key > old)
                        sift_up(i);
                    else
                        sift_down(i);
                }

                // 删除任意元素
                void erase(int id) {
                    int i = pos[id];
                    if (i == -1) return;

                    int last = h.size() - 1;
                    swap_node(i, last);
                    pos[h[last].id] = -1;
                    h.pop_back();

                    if (i < h.size()) {
                        sift_up(i);
                        sift_down(i);
                    }
                }
            };
            tbb::concurrent_hash_map<AddressType, std::string> m_buffer;
            IndexedHeap m_heap;
            std::queue<int> m_freeIds;
            std::mutex m_heapMutex;
            int size;
            int vecSize;
            int capacity;
            int nowMaxId;

            public:
                MergeBuffer(int capacity, int vecSize): m_heap(capacity / vecSize + 1), size(0), vecSize(vecSize), capacity(capacity), nowMaxId(0) {}

                ~MergeBuffer() {}

                

        };

        class CompactionJob : public Helper::ThreadPool::Job
        {
        private:
            LeoFSIO* m_LeoFSIO;

        public:
            CompactionJob(LeoFSIO* LeoFSIO): m_LeoFSIO(LeoFSIO) {}

            ~CompactionJob() {}

            inline void exec(IAbortOperation* p_abort) override {
                m_LeoFSIO->ForceCompaction();
            }
        };

        class LRUCache {
            int capacity;   // Page Num
            int size;
            std::list<SizeType> keys;  // Page Address
            std::unordered_map<SizeType, std::pair<std::string, std::list<SizeType>::iterator>> cache;    // Page Address -> Page Address in Cache
            std::mutex mu;
            int64_t queries;
            int64_t hits;

        public:
            LRUCache(int capacity) {
                this->capacity = capacity;
                this->size = 0;
                this->queries = 0;
                this->hits = 0;
            }

            bool get(SizeType key, void* value) {
                mu.lock();
                queries++;
                auto it = cache.find(key);
                if (it == cache.end()) {
                    mu.unlock();
                    return false;  // 如果键不存在，返回 -1
                }
                // 更新访问顺序，将该键移动到链表头部
                memcpy(value, it->second.first.data(), it->second.first.size());
                keys.splice(keys.begin(), keys, it->second.second);
                it->second.second = keys.begin();
                hits++;
                mu.unlock();
                return true;
            }

            bool put(SizeType key, void* value, int put_size) {
                mu.lock();
                auto it = cache.find(key);
                if (it != cache.end()) {
                    if (put_size > capacity) {
                        size -= it->second.first.size();
                        keys.erase(it->second.second);
                        cache.erase(it);
                        mu.unlock();
                        return false;
                    }
                    auto keys_it = it->second.second;
                    keys.splice(keys.begin(), keys, keys_it);
                    it->second.second = keys.begin();
                    keys_it = keys.begin();
                    auto delta_size = put_size - it->second.first.size();
                    while ((capacity - size) < delta_size && (keys.size() > 1)) {
                        auto last = keys.back();
                        auto it = cache.find(last);
                        size -= it->second.first.size();
                        cache.erase(it);
                        keys.pop_back();
                    }
                    it->second.first.resize(put_size);
                    memcpy(it->second.first.data(), value, put_size);
                    size += delta_size;
                    mu.unlock();
                    return true;
                }
                if (put_size > capacity) {
                    mu.unlock();
                    return false;
                }
                while (put_size > (capacity - size) && (!keys.empty())) {
                    auto last = keys.back();
                    auto it = cache.find(last);
                    size -= it->second.first.size();
                    cache.erase(it);
                    keys.pop_back();
                }
                auto keys_it = keys.insert(keys.begin(), key);
                cache.insert({key, {std::string((char*)value, put_size), keys_it}});
                size += put_size;
                mu.unlock();
                return true;
            }

            bool del(SizeType key) {
                mu.lock();
                auto it = cache.find(key);
                if (it == cache.end()) {
                    mu.unlock();
                    return false;  // 如果键不存在，返回 false
                }
                size -= it->second.first.size();
                keys.erase(it->second.second);
                cache.erase(it);
                mu.unlock();
                return true;
            }

            bool merge(SizeType key, void* value, AddressType merge_size) {
                mu.lock();
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge size: %lld\n", merge_size);
                auto it = cache.find(key);
                if (it == cache.end()) {
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge key not found\n");
                    mu.unlock();
                    return false;  // 如果键不存在，返回 false
                }
                if (merge_size + it->second.first.size() > capacity) {
                    size -= it->second.first.size();
                    keys.erase(it->second.second);
                    cache.erase(it);
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge size exceeded\n");
                    mu.unlock();
                    return false;
                }
                keys.splice(keys.begin(), keys, it->second.second);
                it->second.second = keys.begin();
                while((capacity - size) < merge_size && (keys.size() > 1)) {
                    auto last = keys.back();
                    auto it = cache.find(last);
                    size -= it->second.first.size();
                    cache.erase(it);
                    keys.pop_back();
                }
                it->second.first.append((char*)value, merge_size);
                size += merge_size;
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge success\n");
                mu.unlock();
                return true;
            }
            
            std::pair<int64_t, int64_t> get_stat() {
                return {queries, hits};
            }
        }; 

        class ShardedLRUCache {
            int shards;
            std::vector<LRUCache*> caches;
            SizeType hash(SizeType key) const {
                return key % shards;
            }
        public:
            ShardedLRUCache(int shards, int capacity) : shards(shards) {
                caches.resize(shards);
                for (int i = 0; i < shards; i++) {
                    caches[i] = new LRUCache(capacity / shards);
                }
                if (capacity % shards != 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "LRUCache: capacity is not divisible by shards\n");
                }
            }

            ~ShardedLRUCache() {
                for (int i = 0; i < shards; i++) {
                    delete caches[i];
                }
            }

            bool get(SizeType key, void* value) {
                return caches[hash(key)]->get(key, value);
            }

            bool put(SizeType key, void* value, SizeType put_size) {
                return caches[hash(key)]->put(key, value, put_size);
            }

            bool del(SizeType key) {
                return caches[hash(key)]->del(key);
            }

            bool merge(SizeType key, void* value, AddressType merge_size) {
                return caches[hash(key)]->merge(key, value, merge_size);
            }

            std::pair<int64_t, int64_t> get_stat() {
                int64_t queries = 0, hits = 0;
                for (int i = 0; i < shards; i++) {
                    auto stat = caches[i]->get_stat();
                    queries += stat.first;
                    hits += stat.second;
                }
                return {queries, hits};
            }
        };

    public:
        LeoFSIO(const char* filePath, SizeType blockSize, SizeType capacity, SizeType postingBlocks, SizeType bufferSize = 1024, int batchSize = 64, bool recovery = false, int compactionThreads = 1, Options *opt = nullptr) {
            // TODO: 后面还得再看看，可能需要修改
            m_mappingPath = std::string(filePath);
            m_blockLimit = postingBlocks + 1;
            m_bufferLimit = bufferSize;
            m_pOpt = opt;
            m_pBlockController.SetOpt(opt);
            m_mergeCount = 0;
            m_mergeSize = 0;
            m_putBlockCount = 0;
            m_putCount = 0;

            if (!m_pOpt) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "LeoFSIO: No options provided!\n");
            }
            const char* LeoFSUseLock = getenv(kLeoFSUseLock);
            if(LeoFSUseLock) {
                if(strcmp(LeoFSUseLock, "True") == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO: Using lock\n");
                    m_LeoFSUseLock = true;
                    const char* LeoFSLockSize = getenv(kLeoFSLockSize);
                    if(LeoFSLockSize) {
                        m_LeoFSLockSize = atoi(LeoFSLockSize);
                    }
                    m_rwMutex = std::vector<std::shared_mutex>(m_LeoFSLockSize);
                }
                else {
                    m_LeoFSUseLock = false;
                }
            }
            else {
                m_LeoFSUseLock = false;
            }
            const char* LeoFSUseCache = getenv(kLeoFSUseCache);
            if (LeoFSUseCache) {
                if (strcmp(LeoFSUseCache, "True") == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO: Using cache\n");
                    m_LeoFSUseCache = true;
                }
                else {
                    m_LeoFSUseCache = false;
                }
            }
            if (m_LeoFSUseCache) {
                const char* LeoFSCacheSize = getenv(kLeoFSCacheSize);
                const char* LeoFSCacheShards = getenv(kLeoFSCacheShards);
                int capacity = kSsdLeoFSDefaultCacheSize;
                int shards = kSsdLeoFSDefaultCacheShards;
                if(LeoFSCacheSize) {
                    capacity = atoi(LeoFSCacheSize);
                } 
                if(LeoFSCacheShards) {
                    shards = atoi(LeoFSCacheShards);
                }
                m_pShardedLRUCache = new ShardedLRUCache(shards, capacity);
            }

            const char* LeoFSUseAsync = getenv(kLeoFSUseAsync);
            if (LeoFSUseAsync) {
                if (strcmp(LeoFSUseAsync, "True") == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO: Using async\n");
                    m_LeoFSUseAsync = true;
                }
                else {
                    m_LeoFSUseAsync = false;
                }
            }

            if (recovery) {
                m_mappingPath += "_blockmapping";
                Load(m_mappingPath, blockSize, capacity);
            } else if(fileexists(m_mappingPath.c_str())) {
                Load(m_mappingPath, blockSize, capacity);
            } else {
                m_pBlockMapping.Initialize(0, 1, blockSize, capacity);
            }
            // m_writeCount.resize(m_pBlockMapping.R());
            // m_readCount.resize(m_pBlockMapping.R());
            // m_mergeCount.resize(m_pBlockMapping.R());

            for (int i = 0; i < bufferSize; i++) {
                m_buffer.push((uintptr_t)(new AddressType[m_blockLimit]));
            }
            m_compactionThreadPool = std::make_shared<Helper::ThreadPool>();
            m_compactionThreadPool->init(compactionThreads);
            if (recovery) {
                if (m_pBlockController.Recovery(std::string(filePath), batchSize) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Recover LeoFSIO!\n");
                    exit(0);
                }
            } else if (!m_pBlockController.Initialize(batchSize)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Initialize LeoFSIO!\n");
                exit(0);
            }

            if (opt && opt->m_useBufferedWrite) {
                m_pBlockController.SetWriteBuffer(opt);
                m_useBufferedWrite = true;
            }
            m_shutdownCalled = false;
        }

        ~LeoFSIO() {
            ShutDown();
        }

        void SetOpt(Options *opt) {
            m_pBlockController.SetOpt(opt);
            m_pOpt = opt;
        }

        void ShutDown() override {
            if (m_shutdownCalled) {
                return;
            }
            if (!m_mappingPath.empty()) Save(m_mappingPath);
            // TODO: 这里是不是应该加锁？
            for (int i = 0; i < m_pBlockMapping.R(); i++) {
                if (At(i) != 0xffffffffffffffff) delete[]((AddressType*)At(i));
            }
            while (!m_buffer.empty()) {
                uintptr_t ptr;
                if (m_buffer.try_pop(ptr)) delete[]((AddressType*)ptr);
            }
            m_pBlockController.ShutDown();
            if (m_LeoFSUseCache) {
                delete m_pShardedLRUCache;
            }
            m_shutdownCalled = true;
        }

        inline uintptr_t& At(SizeType key) {
            return *(m_pBlockMapping[key]);
        }

        ErrorCode Get(SizeType key, ByteArray& value) {
            auto get_begin_time = std::chrono::high_resolution_clock::now();
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].lock_shared();
            }
            SizeType r;
            if (m_LeoFSUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;
            
            if (m_LeoFSUseCache) {
                auto size = ((AddressType*)At(key))[0];
                std::uint8_t* outdata = new std::uint8_t[size];
                if (m_pShardedLRUCache->get(key, outdata)) {
                    value.Set(outdata, size, false);
                    if (m_LeoFSUseLock) {
                        m_rwMutex[hash(key)].unlock_shared();
                    }
                    auto get_end_time = std::chrono::high_resolution_clock::now();
                    get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
                    return ErrorCode::Success;
                }
                delete[] outdata;
            }

            // if (m_pBlockController.ReadBlocks((AddressType*)At(key), value)) {
            //     return ErrorCode::Success;
            // }
            auto begin_time = std::chrono::high_resolution_clock::now();
            auto result = m_pBlockController.ReadBlocks((AddressType*)At(key), value);
            auto end_time = std::chrono::high_resolution_clock::now();
            read_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();
            get_times_vec[id]++;
            if (m_LeoFSUseCache) {
                m_pShardedLRUCache->put(key, value.Data(), value.Length());
            }
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].unlock_shared();
            }
            auto get_end_time = std::chrono::high_resolution_clock::now();
            get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Get(SizeType key, std::string* value) override {
            auto get_begin_time = std::chrono::high_resolution_clock::now();
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].lock_shared();
            }
            SizeType r;
            if (m_LeoFSUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;

            // m_readCount[key]++;
            if (m_LeoFSUseCache) {
                auto size = ((AddressType*)At(key))[0];
                value->resize(size);
                if (m_pShardedLRUCache->get(key, value->data())) {
                    if (m_LeoFSUseLock) {
                        m_rwMutex[hash(key)].unlock_shared();
                    }
                    auto get_end_time = std::chrono::high_resolution_clock::now();
                    get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
                    return ErrorCode::Success;
                }
            }   
            
            // if (m_pBlockController.ReadBlocks((AddressType*)At(key), value)) {
            //     return ErrorCode::Success;
            // }
            auto begin_time = std::chrono::high_resolution_clock::now();
            bool result;
            if (m_useBufferedWrite) {
                result = m_pBlockController.BufferedReadBlocks((AddressType*)At(key), value);
            }
            else {
                result = m_pBlockController.NewReadBlocks((AddressType*)At(key), value);
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            read_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();
            get_times_vec[id]++;
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].unlock_shared();
            }
            auto get_end_time = std::chrono::high_resolution_clock::now();
            get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Get(SizeType key, std::string* value, const std::chrono::microseconds &timeout) {
            auto get_begin_time = std::chrono::high_resolution_clock::now();
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].lock_shared();
            }
            SizeType r;
            if (m_LeoFSUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;

            // m_readCount[key]++;
            if (m_LeoFSUseCache) {
                auto size = ((AddressType*)At(key))[0];
                value->resize(size);
                if (m_pShardedLRUCache->get(key, value->data())) {
                    if (m_LeoFSUseLock) {
                        m_rwMutex[hash(key)].unlock_shared();
                    }
                    auto get_end_time = std::chrono::high_resolution_clock::now();
                    get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
                    return ErrorCode::Success;
                }
            }
            
            
            // if (m_pBlockController.ReadBlocks((AddressType*)At(key), value)) {
            //     return ErrorCode::Success;
            // }
            auto begin_time = std::chrono::high_resolution_clock::now();
            bool result;
            if (m_useBufferedWrite) {
                result = m_pBlockController.BufferedReadBlocks((AddressType*)At(key), value, timeout);
            }
            else {
                result = m_pBlockController.NewReadBlocks((AddressType*)At(key), value, timeout);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            read_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();
            get_times_vec[id]++;
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].unlock_shared();
            }
            auto get_end_time = std::chrono::high_resolution_clock::now();
            get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Get(const std::string& key, std::string* value) override {
            return Get(std::stoi(key), value);
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
            std::vector<AddressType*> blocks;
            std::set<int> lock_keys;
            if (m_LeoFSUseLock) {
                // 这里要去重？
                // for (SizeType key : keys) {
                //     lock_keys.insert(hash(key));
                // }
                for (SizeType key : keys) {
                    m_rwMutex[hash(key)].lock_shared();
                }
            }
            SizeType r;
            values->resize(keys.size());
            int i = 0;
            for (SizeType key : keys) {
                // m_readCount[key]++;
                if (m_LeoFSUseLock) {
                    m_updateMutex.lock_shared();
                    r = m_pBlockMapping.R();
                    m_updateMutex.unlock_shared();
                }
                else {
                    r = m_pBlockMapping.R();
                }
                if (key < r) {
                    if (m_LeoFSUseCache) {
                        auto size = ((AddressType*)At(key))[0];
                        (*values)[i].resize(size);
                        if (m_pShardedLRUCache->get(key, (*values)[i].data())) {
                            blocks.push_back(nullptr);
                        }
                        else {
                            blocks.push_back((AddressType*)At(key));
                        }
                    } else {
                        blocks.push_back((AddressType*)At(key));
                    }
                    
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to read key:%d total key number:%d\n", key, r);
                }
                i++;
            }
            // if (m_pBlockController.ReadBlocks(blocks, values, timeout)) return ErrorCode::Success;
            // auto result = m_pBlockController.ReadBlocks(blocks, values, timeout);
            bool result;
            if (m_useBufferedWrite) {
                result = m_pBlockController.BufferedReadBlocks(blocks, values, timeout);
            }
            else if (m_LeoFSUseAsync) {
                result = m_pBlockController.ReadBlocksAsync(blocks, values, timeout);
            }
            else {
                result = m_pBlockController.NewReadBlocks(blocks, values, timeout);
            }
            
            if (m_LeoFSUseLock) {
                for (SizeType key : keys) {
                    m_rwMutex[hash(key)].unlock_shared();
                }
            }
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) override {
            std::vector<SizeType> int_keys;
            for (const auto& key : keys) {
                int_keys.push_back(std::stoi(key));
            }
            return MultiGet(int_keys, values, timeout);
        }

        ErrorCode Scan(const SizeType start_key, const int record_count, std::vector<ByteArray> &values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
            std::vector<SizeType> keys;
            std::vector<AddressType*> blocks;
            SizeType curr_key = start_key;
            while(keys.size() < record_count && curr_key < m_pBlockMapping.R()) {
                if (m_LeoFSUseLock) {
                    m_rwMutex[hash(curr_key)].lock_shared();
                }
                if (At(curr_key) == 0xffffffffffffffff) {
                    if (m_LeoFSUseLock) {
                        m_rwMutex[hash(curr_key)].unlock_shared();
                    }
                    curr_key++;
                    continue;
                }
                keys.push_back(curr_key);
                blocks.push_back((AddressType*)At(curr_key));
                curr_key++;
            }
            auto result = m_pBlockController.ReadBlocks(blocks, values, timeout);
            if (m_LeoFSUseLock) {
                for (auto key : keys) {
                    m_rwMutex[hash(key)].unlock_shared();
                }
            }
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Put(SizeType key, const std::string& value) override {
            int blocks = ((value.size() + PageSize - 1) >> PageSizeEx);
            m_putBlockCount += blocks;
            m_putCount++;
            if (blocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to put key:%d value:%lld since value too long!\n", key, value.size());
                return ErrorCode::Fail;
            }
            // 计算是否需要更多的Mapping块
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            int delta;
            if (m_LeoFSUseLock) {
                m_updateMutex.lock_shared();
                delta = key + 1 - m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                delta = key + 1 - m_pBlockMapping.R();
            }
            if (delta > 0) {
                    // std::lock_guard<std::mutex> lock(m_updateMutex);
                m_updateMutex.lock();
                delta = key + 1 - m_pBlockMapping.R();
                if (delta > 0) {
                    m_pBlockMapping.AddBatch(delta);
                    // m_readCount.resize(m_readCount.size() + delta, 0);
                    // m_writeCount.resize(m_writeCount.size() + delta, 0);
                    // m_mergeCount.resize(m_mergeCount.size() + delta, 0);
                }
                m_updateMutex.unlock();
            }
            // m_writeCount[key]++;

            if (m_LeoFSUseCache) {
                m_pShardedLRUCache->put(key, (void *)(value.data()), value.size());
            }

            // 如果这个key还没有分配过Mapping块，就分配一组
            if (At(key) == 0xffffffffffffffff) {
                // m_buffer里有多的块就直接用，没有就new一组
                if (m_buffer.unsafe_size() > m_bufferLimit) {
                    uintptr_t tmpblocks;
                    while (!m_buffer.try_pop(tmpblocks));
                    At(key) = tmpblocks;
                }
                else {
                    At(key) = (uintptr_t)(new AddressType[m_blockLimit]);
                }
                // 块地址列表里的0号元素代表数据大小，将其设为-1
                memset((AddressType*)At(key), -1, sizeof(AddressType) * m_blockLimit);
            }
            int64_t* postingSize = (int64_t*)At(key);
            // postingSize小于0说明是新分配的Mapping块，直接获取磁盘块，写入数据
            if (*postingSize < 0) {
                m_pBlockController.GetBlocks(postingSize + 1, blocks);
                if (m_useBufferedWrite) {
                    m_pBlockController.BufferedWriteBlocks(postingSize + 1, blocks, value);
                }
                else {
                    m_pBlockController.NewWriteBlocks(postingSize + 1, blocks, value);
                }
                *postingSize = value.size();
            }
            else {
                uintptr_t tmpblocks;
                // 从buffer里拿一组Mapping块，一会再还一组回去
                while (!m_buffer.try_pop(tmpblocks));
                // 获取一组新的磁盘块，直接写入数据
                // 为保证Checkpoint的效果，这里必须分配新的块进行写入
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1, blocks);
                if (m_useBufferedWrite) {
                    m_pBlockController.BufferedWriteBlocks((AddressType*)tmpblocks + 1, blocks, value);
                }
                else {
                    m_pBlockController.NewWriteBlocks((AddressType*)tmpblocks + 1, blocks, value);
                }
                *((int64_t*)tmpblocks) = value.size();

                // 释放原有的块
                m_pBlockController.ReleaseBlocks(postingSize + 1, (*postingSize + PageSize -1) >> PageSizeEx);
                At(key) = tmpblocks;
                m_buffer.push((uintptr_t)postingSize);
            }
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
            return ErrorCode::Success;
        }

        ErrorCode Put(SizeType key, const ByteArray& value) {
            int blocks = ((value.Length() + PageSize - 1) >> PageSizeEx);
            if (blocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to put key:%d value:%lld since value too long!\n", key, value.Length());
                return ErrorCode::Fail;
            }
            // 计算是否需要更多的Mapping块
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            int delta;
            if (m_LeoFSUseLock) {
                m_updateMutex.lock_shared();
                delta = key + 1 - m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                delta = key + 1 - m_pBlockMapping.R();
            }
            if (delta > 0) {
                    // std::lock_guard<std::mutex> lock(m_updateMutex);
                m_updateMutex.lock();
                delta = key + 1 - m_pBlockMapping.R();
                if (delta > 0) {
                    m_pBlockMapping.AddBatch(delta);
                }
                m_updateMutex.unlock();
            }

            if (m_LeoFSUseCache) {
                m_pShardedLRUCache->put(key, (void *)(value.Data()), value.Length());
            }

            // 如果这个key还没有分配过Mapping块，就分配一组
            if (At(key) == 0xffffffffffffffff) {
                // m_buffer里有多的块就直接用，没有就new一组
                if (m_buffer.unsafe_size() > m_bufferLimit) {
                    uintptr_t tmpblocks;
                    while (!m_buffer.try_pop(tmpblocks));
                    At(key) = tmpblocks;
                }
                else {
                    At(key) = (uintptr_t)(new AddressType[m_blockLimit]);
                }
                // 块地址列表里的0号元素代表数据大小，将其设为-1
                memset((AddressType*)At(key), -1, sizeof(AddressType) * m_blockLimit);
            }
            int64_t* postingSize = (int64_t*)At(key);
            // postingSize小于0说明是新分配的Mapping块，直接获取磁盘块，写入数据
            if (*postingSize < 0) {
                m_pBlockController.GetBlocks(postingSize + 1, blocks);
                m_pBlockController.WriteBlocks(postingSize + 1, blocks, value);
                *postingSize = value.Length();
            }
            else {
                uintptr_t tmpblocks;
                // 从buffer里拿一组Mapping块，一会再还一组回去
                while (!m_buffer.try_pop(tmpblocks));
                // 获取一组新的磁盘块，直接写入数据
                // 为保证Checkpoint的效果，这里必须分配新的块进行写入
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1, blocks);
                m_pBlockController.WriteBlocks((AddressType*)tmpblocks + 1, blocks, value);
                *((int64_t*)tmpblocks) = value.Length();

                // 释放原有的块
                m_pBlockController.ReleaseBlocks(postingSize + 1, (*postingSize + PageSize -1) >> PageSizeEx);
                At(key) = tmpblocks;
                m_buffer.push((uintptr_t)postingSize);
            }
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
            return ErrorCode::Success;
        }

        ErrorCode Put(const std::string &key, const std::string& value) override {
            return Put(std::stoi(key), value);
        }

        ErrorCode Merge(SizeType key, const std::string& value) {
            m_mergeCount++;
            m_mergeSize += value.size();
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            SizeType r;
            if (m_LeoFSUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Key range error: key: %d, mapping size: %d\n", key, r);
                if (m_LeoFSUseLock) {
                    m_rwMutex[hash(key)].unlock();
                }
                return ErrorCode::Fail;
            }
            
            int64_t* postingSize = (int64_t*)At(key);

            // m_mergeCount[key]++;
            if (m_LeoFSUseCache) {
                m_pShardedLRUCache->merge(key, (void *)(value.data()), value.size());
            }

            auto newSize = *postingSize + value.size();
            int newblocks = ((newSize + PageSize - 1) >> PageSizeEx);
            if (newblocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failt to merge key:%d value:%lld since value too long!\n", key, newSize);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Origin Size: %lld, merge size: %lld\n", *postingSize, value.size());
                if (m_LeoFSUseLock) {
                    m_rwMutex[hash(key)].unlock();
                }
                return ErrorCode::Fail;
            }

            auto sizeInPage = (*postingSize) % PageSize;    // 最后一个块的实际大小
            int oldblocks = (*postingSize >> PageSizeEx);
            int allocblocks = newblocks - oldblocks;
            // 最后一个块没有写满的话，需要先读出来，然后拼接新的数据，再写回去
            if (sizeInPage != 0) {
                std::string newValue;
                AddressType readreq[] = { sizeInPage, *(postingSize + 1 + oldblocks) };
                if (m_useBufferedWrite) {
                    m_pBlockController.BufferedReadBlocks(readreq, &newValue);
                }
                else {
                    m_pBlockController.NewReadBlocks(readreq, &newValue);
                }
                newValue += value;

                uintptr_t tmpblocks;
                while (!m_buffer.try_pop(tmpblocks));
                memcpy((AddressType*)tmpblocks, postingSize, sizeof(AddressType) * (oldblocks + 1));
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1 + oldblocks, allocblocks);
                if (m_useBufferedWrite) {
                    m_pBlockController.BufferedWriteBlocks((AddressType*)tmpblocks + 1 + oldblocks, allocblocks, newValue);
                }
                else {
                    m_pBlockController.NewWriteBlocks((AddressType*)tmpblocks + 1 + oldblocks, allocblocks, newValue);
                }
                *((int64_t*)tmpblocks) = newSize;

                // 这里也是为了保证Checkpoint，所以将原本没用满的块释放，分配一个新的
                m_pBlockController.ReleaseBlocks(postingSize + 1 + oldblocks, 1);
                At(key) = tmpblocks;
                m_buffer.push((uintptr_t)postingSize);
            }
            else {  // 否则直接分配一组块接在后面
                m_pBlockController.GetBlocks(postingSize + 1 + oldblocks, allocblocks);
                if (m_useBufferedWrite) {
                    m_pBlockController.BufferedWriteBlocks(postingSize + 1 + oldblocks, allocblocks, value);
                }
                else {
                    m_pBlockController.NewWriteBlocks(postingSize + 1 + oldblocks, allocblocks, value);
                }
                *postingSize = newSize;
            }
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
            return ErrorCode::Success;
        }

        ErrorCode Merge(const std::string &key, const std::string& value) {
            return Merge(std::stoi(key), value);
        }

        ErrorCode Delete(SizeType key) override {
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            SizeType r;
            if (m_LeoFSUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;

            if (m_LeoFSUseCache) {
                m_pShardedLRUCache->del(key);
            }

            int64_t* postingSize = (int64_t*)At(key);
            if (*postingSize < 0) {
                if (m_LeoFSUseLock) {
                    m_rwMutex[hash(key)].unlock();
                }
                return ErrorCode::Fail;
            }

            int blocks = ((*postingSize + PageSize - 1) >> PageSizeEx);
            m_pBlockController.ReleaseBlocks(postingSize + 1, blocks);
            m_buffer.push((uintptr_t)postingSize);
            At(key) = 0xffffffffffffffff;
            if (m_LeoFSUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
            return ErrorCode::Success;
        }

        ErrorCode Delete(const std::string &key) {
            return Delete(std::stoi(key));
        }

        void ForceCompaction() {
            Save(m_mappingPath);
        }

        void GetStat() {
            int remainBlocks = m_pBlockController.RemainBlocks();
            int remainGB = (long long)remainBlocks << PageSizeEx >> 30;
            // int remainGB = remainBlocks >> 20 << 2;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Remain %d blocks, totally %d GB\n", remainBlocks, remainGB);
            uint64_t get_times = 0;
            uint64_t get_time = 0;
            uint64_t read_time = 0;
            for (int i = 0; i < get_time_vec.size(); i++) {
                get_times += get_times_vec[i];
                get_time += get_time_vec[i];
                read_time += read_time_vec[i];
            }
            double average_read_time = (double)read_time / get_times;
            double average_get_time = (double)get_time / get_times;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Get times: %llu, get time: %llu us, read time: %llu us\n", get_times, get_time, read_time);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Average read time: %lf us, average get time: %lf us\n", average_read_time, average_get_time);
            if (m_LeoFSUseCache) {
                auto cache_stat = m_pShardedLRUCache->get_stat();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Cache queries: %lld, Cache hits: %lld, Hit rates: %lf\n", cache_stat.first, cache_stat.second, cache_stat.second == 0 ? 0 : (double)cache_stat.second / cache_stat.first);
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg merge size: %lf, merge count: %lld\n", (double)m_mergeSize.load() / m_mergeCount.load(), m_mergeCount.load());
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg put block count: %lf, put count: %lld\n", (double)m_putBlockCount.load() / m_putCount.load(), m_putCount.load());
            m_putBlockCount = 0;
            m_putCount = 0;
            m_pBlockController.IOStatistics();
        }

        // ErrorCode PrintHotColdStat() {
        //     std::string filePath = "hotcold" + std::to_string(m_hotColdStatNo) + ".txt";
        //     m_hotColdStatNo++;
        //     std::ofstream fout(filePath);
        //     if (!fout.is_open()) {
        //         SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Print hot/cold stat: Failed to open file %s\n", filePath.c_str());
        //         return ErrorCode::Fail;
        //     }
        //     fout << "Read Stat:" << std::endl;
        //     for (int i = 0; i < m_readCount.size(); i++) {
        //         fout << i << " " << m_readCount[i] << std::endl;
        //     }
        //     fout << "Write Stat:" << std::endl;
        //     for (int i = 0; i < m_writeCount.size(); i++) {
        //         fout << i << " " << m_writeCount[i] << std::endl;
        //     }
        //     fout << "Merge Stat:" << std::endl;
        //     for (int i = 0; i < m_mergeCount.size(); i++) {
        //         fout << i << " " << m_mergeCount[i] << std::endl;
        //     }
        //     fout.close();
        //     return ErrorCode::Success;
        // }

        // ErrorCode Load(std::string path, SizeType blockSize, SizeType capacity) {
        //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping From %s\n", path.c_str());
        //     // auto ptr = f_createIO();
        //     // if (ptr == nullptr || !ptr->Initialize(path.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
        //     int cid = m_pBlockController.GetCid();
        //     int ldfd = dfs_open(cid, path.c_str(), O_RDONLY, 0644);

        //     SizeType CR, mycols;
        //     // IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&CR);
        //     // IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&mycols);
        //     if (dfs_read(cid, ldfd, (char*)&CR, sizeof(SizeType)) != sizeof(SizeType)) {
        //         return ErrorCode::DiskIOFail;
        //     }
        //     if (dfs_read(cid, ldfd, (char*)&mycols, sizeof(SizeType)) != sizeof(SizeType)) {
        //         return ErrorCode::DiskIOFail;
        //     }
        //     if (mycols > m_blockLimit) m_blockLimit = mycols;

        //     m_pBlockMapping.Initialize(CR, 1, blockSize, capacity);
        //     for (int i = 0; i < CR; i++) {
        //         At(i) = (uintptr_t)(new AddressType[m_blockLimit]);
        //         // IOBINARY(ptr, ReadBinary, sizeof(AddressType) * mycols, (char*)At(i));
        //         if (dfs_read(cid, ldfd, (char*)At(i), sizeof(AddressType) * mycols) != sizeof(AddressType) * mycols) {
        //             return ErrorCode::DiskIOFail;
        //         }
        //     }
        //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping (%d,%d) Finish!\n", CR, mycols);
        //     return ErrorCode::Success;
        // }

        ErrorCode Load(std::string path, SizeType blockSize, SizeType capacity) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping From %s\n", path.c_str());
            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(path.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;

            SizeType CR, mycols;
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&mycols);
            if (mycols > m_blockLimit) m_blockLimit = mycols;

            m_pBlockMapping.Initialize(CR, 1, blockSize, capacity);
            for (int i = 0; i < CR; i++) {
                At(i) = (uintptr_t)(new AddressType[m_blockLimit]);
                IOBINARY(ptr, ReadBinary, sizeof(AddressType) * mycols, (char*)At(i));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping (%d,%d) Finish!\n", CR, mycols);
            return ErrorCode::Success;
        }

        ErrorCode LoadDFS(std::string path, SizeType blockSize, SizeType capacity) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping From %s\n", path.c_str());
            auto ptr = f_createDFSIO();
            if (ptr == nullptr || !ptr->Initialize(-1, m_pOpt->m_leoFSConfigPath.c_str(), path.c_str(), O_RDONLY, 0644)) return ErrorCode::FailedOpenFile;

            SizeType CR, mycols;
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&mycols);
            if (mycols > m_blockLimit) m_blockLimit = mycols;

            m_pBlockMapping.Initialize(CR, 1, blockSize, capacity);
            for (int i = 0; i < CR; i++) {
                At(i) = (uintptr_t)(new AddressType[m_blockLimit]);
                IOBINARY(ptr, ReadBinary, sizeof(AddressType) * mycols, (char*)At(i));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping (%d,%d) Finish!\n", CR, mycols);
            return ErrorCode::Success;
        }

        ErrorCode Save(std::string path) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping To %s\n", path.c_str());
            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(path.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;

            SizeType CR = m_pBlockMapping.R();
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&m_blockLimit);
            std::vector<AddressType> empty(m_blockLimit, 0xffffffffffffffff);
            for (int i = 0; i < CR; i++) {
                if (At(i) == 0xffffffffffffffff) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType) * m_blockLimit, (char*)(empty.data()));
                }
                else {
                    int64_t* postingSize = (int64_t*)At(i);
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType) * m_blockLimit, (char*)postingSize);
                }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping (%d,%d) Finish!\n", CR, m_blockLimit);
            return ErrorCode::Success;
        }

        ErrorCode SaveDFS(std::string path) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping To %s\n", path.c_str());
            auto ptr = f_createDFSIO();
            if (ptr == nullptr || !ptr->Initialize(-1, m_pOpt->m_leoFSConfigPath.c_str(), path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644)) return ErrorCode::FailedCreateFile;

            SizeType CR = m_pBlockMapping.R();
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&m_blockLimit);
            std::vector<AddressType> empty(m_blockLimit, 0xffffffffffffffff);
            for (int i = 0; i < CR; i++) {
                if (At(i) == 0xffffffffffffffff) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType) * m_blockLimit, (char*)(empty.data()));
                }
                else {
                    int64_t* postingSize = (int64_t*)At(i);
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType) * m_blockLimit, (char*)postingSize);
                }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping (%d,%d) Finish!\n", CR, m_blockLimit);
            return ErrorCode::Success;
        }

        bool Initialize(bool debug = false) override {
            if (debug) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Initialize LeoFSIO for new threads\n");
            m_freeIdMutex.lock();
            if (m_freeId.empty()) {
                id = m_maxId++;
            }
            else {
                id = m_freeId.front();
                m_freeId.pop();
            }
            while(read_time_vec.size() <= m_maxId) read_time_vec.push_back(0);
            while(get_time_vec.size() <= m_maxId) get_time_vec.push_back(0);
            while(get_times_vec.size() <= m_maxId) get_times_vec.push_back(0);
            m_freeIdMutex.unlock();
            return m_pBlockController.Initialize(64);
        }

        bool ExitBlockController(bool debug = false) override { 
            if (debug) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Exit LeoFSIO for thread\n");
            m_freeIdMutex.lock();
            m_freeId.push(id);
            m_freeIdMutex.unlock();
            return m_pBlockController.ShutDown(); 
        }

        ErrorCode Checkpoint(std::string prefix) override {
            std::string filename = prefix + "_blockmapping";
            if (m_pOpt->m_saveToLeoFS) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO: saving block mapping to DFS\n");
                auto ret = SaveDFS(filename);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO: saving block mapping to DFS failed\n");
                    return ret;
                }
                ret = m_pBlockController.CheckpointDFS(prefix);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO: saving block controller to DFS failed\n");
                }
                return ret;
            } 
            else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO: saving block mapping\n");
                auto ret = Save(filename);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO: saving block mapping failed\n");
                }
                ret = m_pBlockController.Checkpoint(prefix);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO: saving block controller failed\n");
                }
                return ret;
            }
        }

    private:
        static constexpr const char* kLeoFSUseLock = "SPFRESH_LEOFS_IO_USE_LOCK";
        static constexpr bool kLeoFSDefaultUseLock = false;
        static constexpr const char* kLeoFSLockSize = "SPFRESH_LEOFS_IO_LOCK_SIZE";
        static constexpr int kLeoFSDefaultLockSize = 1024;
        static constexpr const char* kLeoFSUseCache = "SPFRESH_LEOFS_IO_USE_CACHE";
        static constexpr bool kLeoFSDefaultUseCache = false;
        static constexpr const char* kLeoFSCacheSize = "SPFRESH_LEOFS_IO_CACHE_SIZE";
        static constexpr int kSsdLeoFSDefaultCacheSize = 8192 << 10;
        static constexpr const char* kLeoFSCacheShards = "SPFRESH_LEOFS_IO_CACHE_SHARDS";
        static constexpr int kSsdLeoFSDefaultCacheShards = 4;
        static constexpr const char* kLeoFSUseAsync = "SPFRESH_LEOFS_IO_USE_ASYNC";
        static constexpr bool kLeoFSDefaultUseAsync = false;

        static thread_local int id;
        int m_maxId = 0;
        std::queue<int> m_freeId;
        std::mutex m_freeIdMutex;
        std::vector<uint64_t> read_time_vec;
        std::vector<uint64_t> get_time_vec;
        std::vector<uint64_t> get_times_vec;

        bool m_LeoFSUseLock = kLeoFSDefaultUseLock;
        int m_LeoFSLockSize = kLeoFSDefaultLockSize;
        bool m_LeoFSUseCache = kLeoFSDefaultUseCache;
        bool m_LeoFSUseAsync = kLeoFSDefaultUseAsync;
        std::string m_mappingPath;
        SizeType m_blockLimit;
        COMMON::Dataset<uintptr_t> m_pBlockMapping;

        // std::vector<int64_t> m_writeCount;
        // std::vector<int64_t> m_readCount;
        // std::vector<int64_t> m_mergeCount;
        int m_hotColdStatNo = 0;

        SizeType m_bufferLimit;
        tbb::concurrent_queue<uintptr_t> m_buffer;

        std::shared_ptr<Helper::ThreadPool> m_compactionThreadPool;
        BlockController m_pBlockController;
        ShardedLRUCache *m_pShardedLRUCache;

        bool m_shutdownCalled;
        std::shared_mutex m_updateMutex;
        std::vector<std::shared_mutex> m_rwMutex;

        Options *m_pOpt;

        bool m_useBufferedWrite = false;

        std::atomic_int64_t m_mergeSize;
        std::atomic_int64_t m_mergeCount;
        std::atomic_int64_t m_putBlockCount;
        std::atomic_int64_t m_putCount;

        inline int hash(int key) {
            return key % m_LeoFSLockSize;
        }
    };
}
#endif // EXTRA_LEOFS_CONTROLLER_H