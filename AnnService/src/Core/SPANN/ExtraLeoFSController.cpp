#include "inc/Core/SPANN/ExtraLeoFSController.h"

namespace SPTAG::SPANN
{
thread_local struct LeoFSIO::BlockController::IoContext LeoFSIO::BlockController::m_currIoContext;
thread_local int LeoFSIO::BlockController::debug_fd = -1;
thread_local void* LeoFSIO::BlockController::aligned_buf = nullptr;
thread_local int LeoFSIO::id = 0;
thread_local int LeoFSIO::BlockController::cid = -1;
thread_local int LeoFSIO::BlockController::fd = -1;
#ifdef USE_ASYNC_IO
thread_local uint64_t LeoFSIO::BlockController::iocp = 0;
thread_local int LeoFSIO::BlockController::id = 0;
#endif
std::chrono::high_resolution_clock::time_point LeoFSIO::BlockController::m_startTime;
int LeoFSIO::BlockController::m_ssdInflight = 0;
// std::atomic<int> LeoFSIO::BlockController::m_ioCompleteCount(0);
// int LeoFSIO::BlockController::fd = -1;
// int LeoFSIO::BlockController::cid = -1;
char* LeoFSIO::BlockController::filePath = new char[1024];
std::unique_ptr<char[]> LeoFSIO::BlockController::m_memBuffer;

void* LeoFSIO::BlockController::InitializeLeoFS(void* args) {
    LeoFSIO::BlockController* ctrl = (LeoFSIO::BlockController *)args;
    memset(filePath, 0, 1024);
    const char* LeoFSConfigPath = getenv(kLeoFSConfigPath);
    if(LeoFSConfigPath) {
        auto begin = std::chrono::high_resolution_clock::now();
        cid = dfs_connect_config(LeoFSConfigPath);
        auto end = std::chrono::high_resolution_clock::now();
        if (cid < 0) {
            fprintf(stderr, "LeoFSIO::BlockController::InitializeLeoFS failed: dfs_connect_config failed\n");
            ctrl->m_LeoFSThreadStartFailed = true;
        }
        else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: dfs_connect_config success, cid=%d, time elapsed: %lld us\n", cid, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        }
    } else {
        fprintf(stderr, "LeoFSIO::BlockController::InitializeLeoFS failed: No LeoFSConfigPath\n");
        ctrl->m_LeoFSThreadStartFailed = true;
    }
    struct stat st;
    const char* LeoFSPath = getenv(kLeoFSPath);
    auto fileSize = kSsdImplMaxNumBlocks << PageSizeEx;
    if(LeoFSPath) {
        strcpy(filePath, LeoFSPath);
    }
    else {
        fprintf(stderr, "LeoFSIO::BlockController::InitializeLeoFS failed: No filePath\n");
        ctrl->m_LeoFSThreadStartFailed = true;
        fd = -1;
        // strcpy(filePath, "/home/lml/SPFreshTest/testfile");
    }
    if (cid >= 0) {
        if(dfs_stat(cid, filePath, &st) != 0) {
            fd = dfs_open(cid, filePath, O_CREAT | O_WRONLY, 0666);
            if(fd < 0) {
                fprintf(stderr, "LeoFSIO::BlockController::InitializeLeoFS failed\n");
                // return nullptr;
                ctrl->m_LeoFSThreadStartFailed = true;
            }
            else {
                // if (fallocate(fd, 0, 0, fileSize) == -1) {
                //     fprintf(stderr, "LeoFSIO::BlockController::InitializeLeoFS failed: fallocate failed\n");
                //     ctrl->m_LeoFSThreadStartFailed = true;
                //     fd = -1;
                // } else {
                //     close(fd);
                //     fd = open(filePath, O_RDWR | O_DIRECT);
                //     if (fd == -1) {
                //         auto err_str = strerror(errno);
                //         fprintf(stderr, "open failed: %s\n", err_str);
                //         ctrl->m_LeoFSThreadStartFailed = true;
                //     } else {
                //         SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: file %s created, fd=%d\n", filePath, fd);
                //     }
                // }
                dfs_close(cid, fd);
                fd = dfs_open(cid, filePath, O_RDWR | O_DIRECT, 0666);
                if (fd < 0) {
                    auto err_str = dfs_errno(cid);
                    fprintf(stderr, "open failed: %s\n", err_str);
                    ctrl->m_LeoFSThreadStartFailed = true;
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: file %s opened, fd=%d\n", filePath, fd);
                }
            }
        }
        else {
            // auto actualFileSize = st.st_blocks * st.st_blksize;
            // if(actualFileSize < fileSize) {
            //     fd = open(filePath, O_CREAT | O_WRONLY, 0666);
            //     if(fd == -1) {
            //         fprintf(stderr, "LeoFSIO::BlockController::InitializeLeoFS failed\n");
            //         // return nullptr;
            //         ctrl->m_LeoFSThreadStartFailed = true;
            //     }
            //     else {
            //         if (fallocate(fd, 0, 0, fileSize) == -1) {
            //             fprintf(stderr, "LeoFSIO::BlockController::InitializeLeoFS failed: fallocate failed\n");
            //             ctrl->m_LeoFSThreadStartFailed = true;
            //             fd = -1;
            //         } else {
            //             close(fd);
            //             fd = open(filePath, O_RDWR | O_DIRECT);
            //             if (fd == -1) {
            //                 auto err_str = strerror(errno);
            //                 fprintf(stderr, "open failed: %s\n", err_str);
            //                 ctrl->m_LeoFSThreadStartFailed = true;
            //             } else {
            //                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: file %s created, fd=%d\n", filePath, fd);
            //             }
            //         }
            //     }
            // } else {
            //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: file has been created with enough space.\n", filePath, fd);
            //     fd = open(filePath, O_RDWR | O_DIRECT);
            //     if (fd == -1) {
            //         auto err_str = strerror(errno);
            //         fprintf(stderr, "open failed: %s\n", err_str);
            //         ctrl->m_LeoFSThreadStartFailed = true;
            //     } else {
            //         SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: file %s opened, fd=%d\n", filePath, fd);
            //     }
            // }
            fd = dfs_open(cid, filePath, O_RDWR | O_DIRECT, 0666);
            if (fd < 0) {
                auto err_str = dfs_errno(cid);
                fprintf(stderr, "open failed: %s\n", err_str);
                ctrl->m_LeoFSThreadStartFailed = true;
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: file %s opened, fd=%d\n", filePath, fd);
            }
        }
    }
    
    const char* LeoFSDepth = getenv(kLeoFSDepth);
    if (LeoFSDepth) ctrl->m_ssdLeoFSDepth = atoi(LeoFSDepth);
    const char* LeoFSThreadNum = getenv(kLeoFSThreadNum);
    if (LeoFSThreadNum) ctrl->m_ssdLeoFSThreadNum = atoi(LeoFSThreadNum);
    const char* LeoFSAlignment = getenv(kLeoFSAlignment);
    if (LeoFSAlignment) ctrl->m_ssdLeoFSAlignment = atoi(LeoFSAlignment);

    // ctrl->m_prwThreadPool = new LeoFSIO::BlockController::RWThreadPool();
    // ctrl->m_prwThreadPool->init(ctrl->m_ssdLeoFSThreadNum);

    ctrl->m_batchReadTimes = 0;
    ctrl->m_batchReadTimeouts = 0;

    if(ctrl->m_LeoFSThreadStartFailed == false) {
        ctrl->m_LeoFSThreadReady = true;
        // std::lock_guard<std::mutex> lock(ctrl->m_uniqueResourceMutex);
        // m_ssdInflight = 0;
    }

    if (cid >= 0) {
        if (fd >= 0) 
            dfs_close(cid, fd);
        dfs_disconnect(cid);
    }
    pthread_exit(NULL);
}

bool LeoFSIO::BlockController::Initialize(int batchSize) {
    std::lock_guard<std::mutex> lock(m_initMutex);
    m_numInitCalled++;

    if(m_numInitCalled == 1) {
        m_batchSize = batchSize;
        m_startTime = std::chrono::high_resolution_clock::now();
        for(AddressType i = 0; i < kSsdImplMaxNumBlocks; i++) {
            m_blockAddresses.push(i);
        }
        pthread_create(&m_LeoFSTid, NULL, &InitializeLeoFS, this);
        while(!m_LeoFSThreadReady && !m_LeoFSThreadStartFailed);
        if(m_LeoFSThreadStartFailed) {
            fprintf(stderr, "LeoFSIO::BlockController::Initialize failed\n");
            return false;
        }
    }

    const char* LeoFSConfigPath = getenv(kLeoFSConfigPath);
    if (!LeoFSConfigPath) {
        fprintf(stderr, "LeoFSIO::BlockController::Initialize failed: LeoFSConfigPath is not set\n");
        return false;
    }
    if (m_numInitCalled == 1) {
        m_LeoFSConfigPath = std::string(LeoFSConfigPath);
    }

    cid = -1;
    fd = -1;
    while(!m_cidFds.empty()) {
        std::pair<int, int> p;
        if(m_cidFds.try_pop(p)) {
            cid = p.first;
            fd = p.second;
            break;
        }
    }
    if (cid < 0) {
        auto begin = std::chrono::high_resolution_clock::now();
        cid = dfs_connect_config(LeoFSConfigPath);
        auto end = std::chrono::high_resolution_clock::now();
        if (cid < 0) {
            fprintf(stderr, "LeoFSIO::BlockController::Initialize failed: dfs_connect_config failed\n");
            return false;
        }
        else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::InitializeLeoFS: dfs_connect_config success, cid=%d, time elapsed: %lld us\n", cid, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        }

        if (fd < 0) {
            fd = dfs_open(cid, filePath, O_RDWR | O_DIRECT, 0666);
            if (fd < 0) {
                auto err_str = dfs_errno(cid);
                fprintf(stderr, "open failed: %s\n", err_str);
                return false;
            }
        }
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
    while(multi_read_time_vec.size() <= id) {
        multi_read_time_vec.push_back(0);
    }
    while(multi_read_times.size() <= id) {
        multi_read_times.push_back(0);
    }
    m_currIoContext.sub_io_requests.resize(m_ssdLeoFSDepth);
    m_currIoContext.in_flight = 0;
    for(auto &sr : m_currIoContext.sub_io_requests) {
        sr.app_buff = nullptr;
        memset(&(sr.myiocb), 0, sizeof(struct dfs_iocb));
        auto buf_ptr = aligned_alloc(m_ssdLeoFSAlignment, PageSize);
        if (buf_ptr == nullptr) {
            fprintf(stderr, "LeoFSIO::BlockController::Initialize failed: aligned_alloc failed\n");
            return false;
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
        return false;
    }
#ifdef USE_FILE_DEBUG
    auto debug_file_name = std::string("/nvme1n1/lml/") + std::to_string(m_numInitCalled) + "_debug.log";
    debug_fd = open(debug_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (debug_fd == -1) {
        fprintf(stderr, "LeoFSIO::BlockController::Initialize failed: open debug file failed\n");
        return false;
    }
#endif
    return true;
}

bool LeoFSIO::BlockController::GetBlocks(AddressType* p_data, int p_size) {
    AddressType currBlockAddress = 0;
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 1";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::GetBlocks: %s\n", strerror(errno));
    }
    fsync(debug_fd);
#endif
    for(int i = 0; i < p_size; i++) {
        while(!m_blockAddresses.try_pop(currBlockAddress));
        p_data[i] = currBlockAddress;
    }
    return true;
}

bool LeoFSIO::BlockController::ReleaseBlocks(AddressType* p_data, int p_size) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 2";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReleaseBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    for(int i = 0; i < p_size; i++) {
        m_blockAddresses_reserve.push(p_data[i]);
    }
    return true;
}

bool LeoFSIO::BlockController::ReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 3";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    p_value->resize(p_data[0]);
    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    auto blockNum = (p_data[0] + PageSize - 1) >> PageSizeEx;
    read_submit_vec[id] += blockNum;
    if (blockNum > m_ssdLeoFSDepth) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: blockNum > m_ssdLeoFSDepth\n");
        return false;
    }

    for (int i = 0; i < blockNum; i++) {
        void *buf = (void*)p_value->data() + currOffset;
        uint64_t real_size = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
        uint64_t offset = p_data[dataIdx] * PageSize;
        // std::cout << "Address: " << p_data[dataIdx] << " Offset: " << offset << " RealSize: " << real_size << std::endl;
        memset(aligned_buf, 0, PageSize);
        int ret = dfs_pread(cid, fd, aligned_buf, real_size, offset);
        if (ret != real_size) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: dfs_pread failed\n");
            return false;
        }
        memcpy(buf, aligned_buf, real_size);
        dataIdx++;
        currOffset += PageSize;
        read_complete_vec[id]++;
    }
    return true;
}

bool LeoFSIO::BlockController::NewReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout) {
    p_value->resize(p_data[0]);
    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    auto blockNum = (p_data[0] + PageSize - 1) >> PageSizeEx;
    if (blockNum == 0) {
        // std::cout << p_data[0] << std::endl;
        return true;
    }
    read_submit_vec[id] += blockNum;
    if (blockNum > m_ssdLeoFSDepth) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: blockNum > m_ssdLeoFSDepth\n");
        return false;
    }

    std::vector<dfs_iocb*> iocbs(blockNum);

    for (int i = 0; i < blockNum; i++) {
        void *buf = (void*)p_value->data() + currOffset;
        uint64_t real_size = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
        uint64_t offset = p_data[dataIdx] * PageSize;
        // std::cout << "Address: " << p_data[dataIdx] << " Offset: " << offset << " RealSize: " << real_size << std::endl;
        auto currSubIo = m_currIoContext.free_sub_io_requests.front();
        m_currIoContext.free_sub_io_requests.pop();
        iocbs[i] = &(currSubIo->myiocb);
        iocbs[i]->aio_nbytes = real_size;
        iocbs[i]->aio_offset = offset;
        currSubIo->app_buff = buf;
        dataIdx++;
        currOffset += PageSize;
        read_complete_vec[id]++;
    }

    auto ret = dfs_multi_pread(cid, fd, blockNum, iocbs.data());
    for (int i = 0; i < blockNum; i++) {
        auto currSubIo = reinterpret_cast<SubIoRequest*>(iocbs[i]->aio_data);
        m_currIoContext.free_sub_io_requests.push(currSubIo);
        if (ret < 0) {
            continue;
        }
        memcpy(currSubIo->app_buff, iocbs[i]->aio_buf, iocbs[i]->aio_nbytes);
        read_complete_vec[id]++;
    }
    return true;
}

bool LeoFSIO::BlockController::BufferedReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout){
    p_value->resize(p_data[0]);
    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    auto blockNum = (p_data[0] + PageSize - 1) >> PageSizeEx;
    if (blockNum == 0) {
        // std::cout << p_data[0] << std::endl;
        return true;
    }
    read_submit_vec[id] += blockNum;
    if (blockNum > m_ssdLeoFSDepth) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: blockNum > m_ssdLeoFSDepth\n");
        return false;
    }

    std::vector<dfs_iocb*> iocbs;
    iocbs.reserve(blockNum);

    for (int i = 0; i < blockNum; i++) {
        void *buf = (void*)p_value->data() + currOffset;
        uint64_t real_size = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
        uint64_t offset = p_data[dataIdx] * PageSize;
        // std::cout << "Address: " << p_data[dataIdx] << " Offset: " << offset << " RealSize: " << real_size << std::endl;

        if (!m_pWriteBuffer->get(offset, buf, real_size)) {
            auto currSubIo = m_currIoContext.free_sub_io_requests.front();
            m_currIoContext.free_sub_io_requests.pop();
            iocbs.push_back(&(currSubIo->myiocb));
            currSubIo->myiocb.aio_nbytes = real_size;
            currSubIo->myiocb.aio_offset = offset;
            currSubIo->app_buff = buf;
        }
        
        dataIdx++;
        currOffset += PageSize;
        read_complete_vec[id]++;
    }

    auto ret = dfs_multi_pread(cid, fd, iocbs.size(), iocbs.data());
    for (int i = 0; i < iocbs.size(); i++) {
        auto currSubIo = reinterpret_cast<SubIoRequest*>(iocbs[i]->aio_data);
        m_currIoContext.free_sub_io_requests.push(currSubIo);
        if (ret < 0) {
            continue;
        }
        memcpy(currSubIo->app_buff, iocbs[i]->aio_buf, iocbs[i]->aio_nbytes);
        read_complete_vec[id]++;
    }
    return true;
}

bool LeoFSIO::BlockController::ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 4";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    auto t1 = std::chrono::high_resolution_clock::now();
    // auto aligned_buf = aligned_alloc(PageSize, PageSize);
    m_batchReadTimes++;
    p_values->resize(p_data.size());
    const int batch_size = m_batchSize;

    for (size_t i = 0; i < p_data.size(); i++) {
        AddressType* p_data_i = p_data[i];
        std::string* p_value = &((*p_values)[i]);
        bool io_failed = false;

        if (p_data_i == nullptr) {
            continue;
        }

        p_value->resize(p_data_i[0]);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;

        while(currOffset < p_data_i[0]) {
            SubIoRequest currSubIo;
            void *buf = (void*)p_value->data() + currOffset;
            uint64_t real_size = (p_data_i[0] - currOffset) < PageSize ? (p_data_i[0] - currOffset) : PageSize;
            uint64_t offset = p_data_i[dataIdx] * PageSize;
            if (offset < 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: offset is negative\n");
                exit(1);
            }
            int posting_id = i;
            memset(aligned_buf, 0, PageSize);
            auto ret = dfs_pread(cid, fd, aligned_buf, real_size, offset);
            if (ret < 0) {
                io_failed = true;
                break;
            }
            memcpy(buf, aligned_buf, real_size);
            read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
            read_complete_vec[id]++;
        }
        if (io_failed) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::ReadBlocks: dfs_pread failed\n");
            p_value->clear();
        }
    }

    return true;
}

bool LeoFSIO::BlockController::NewReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout) {
    auto t1 = std::chrono::high_resolution_clock::now();
    m_batchReadTimes++;
    p_values->resize(p_data.size());
    const int batch_size = m_batchSize;
    std::vector<dfs_iocb*> iocbs(batch_size);
    // std::vector<dfs_io_event> events(batch_size);
    std::vector<SubIoRequest> subIoRequests;
    std::vector<int> subIoRequestCount(p_data.size(), 0);
    subIoRequests.reserve(256);
    for(size_t i = 0; i < p_data.size(); i++) {
        AddressType* p_data_i = p_data[i];
        std::string* p_value = &((*p_values)[i]);

        if (p_data_i == nullptr) {
            continue;
        }

        p_value->resize(p_data_i[0]);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;

        while(currOffset < p_data_i[0]) {
            SubIoRequest currSubIo;
            currSubIo.app_buff = (void*)p_value->data() + currOffset;
            currSubIo.real_size = (p_data_i[0] - currOffset) < PageSize ? (p_data_i[0] - currOffset) : PageSize;
            currSubIo.offset = p_data_i[dataIdx] * PageSize;
            currSubIo.posting_id = i;
            subIoRequests.push_back(currSubIo);
            subIoRequestCount[i]++;
            read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
        }
    }

    multi_read_times[id]++;

    for (int currSubIoStartId = 0; currSubIoStartId < subIoRequests.size(); currSubIoStartId += batch_size) {
        int currSubIoEndId = (currSubIoStartId + batch_size) > subIoRequests.size() ? subIoRequests.size() : currSubIoStartId + batch_size;
        int currSubIoIdx = currSubIoStartId;
        int totalToSubmit = currSubIoEndId - currSubIoStartId;
        int totalSubmitted = 0, totalDone = 0;
        for (int i = 0; i < totalToSubmit; i++) {
            auto currSubIoIdx = currSubIoStartId + i;
            auto currSubIo = m_currIoContext.free_sub_io_requests.front();
            m_currIoContext.free_sub_io_requests.pop();
            currSubIo->app_buff = subIoRequests[currSubIoIdx].app_buff;
            currSubIo->real_size = subIoRequests[currSubIoIdx].real_size;
            currSubIo->posting_id = subIoRequests[currSubIoIdx].posting_id;
            currSubIo->myiocb.aio_lio_opcode = 0; // IO_CMD_PREAD
            currSubIo->myiocb.aio_offset = subIoRequests[currSubIoIdx].offset;
            currSubIo->myiocb.aio_nbytes = subIoRequests[currSubIoIdx].real_size;
            iocbs[i] = &(currSubIo->myiocb);
            currSubIoIdx++;
        }
        auto begin = std::chrono::high_resolution_clock::now();
        int ret = dfs_multi_pread(cid, fd, totalToSubmit, iocbs.data());
        auto end = std::chrono::high_resolution_clock::now();
        multi_read_time_vec[id] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        if (ret >= 0) {
            for (int i = 0; i < totalToSubmit; i++) {
                auto currSubIo = reinterpret_cast<SubIoRequest*>(iocbs[i]->aio_data);
                subIoRequestCount[currSubIo->posting_id]--;
                m_currIoContext.free_sub_io_requests.push(currSubIo);
                memcpy(currSubIo->app_buff, currSubIo->myiocb.aio_buf, currSubIo->real_size);
                read_complete_vec[id]++;
            }
        }
        else {
            for (int i = 0; i < totalToSubmit; i++) {
                auto currSubIo = reinterpret_cast<SubIoRequest*>(iocbs[i]->aio_data);
                m_currIoContext.free_sub_io_requests.push(currSubIo);
            }
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            break;
        }
    }

    bool is_timeout = false;
    for (int i = 0; i < subIoRequestCount.size(); i++) {
        if (subIoRequestCount[i] != 0) {
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks (batch) : timeout\n");
            (*p_values)[i].clear();
            is_timeout = true;
        }
    }
    if (is_timeout) {
        m_batchReadTimeouts++;
    }
    return true;
}

bool LeoFSIO::BlockController::BufferedReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout) {
    auto t1 = std::chrono::high_resolution_clock::now();
    m_batchReadTimes++;
    p_values->resize(p_data.size());
    const int batch_size = m_batchSize;
    std::vector<dfs_iocb*> iocbs;
    // std::vector<dfs_io_event> events(batch_size);
    std::vector<SubIoRequest> subIoRequests;
    std::vector<int> subIoRequestCount(p_data.size(), 0);
    subIoRequests.reserve(256);
    for(size_t i = 0; i < p_data.size(); i++) {
        AddressType* p_data_i = p_data[i];
        std::string* p_value = &((*p_values)[i]);

        if (p_data_i == nullptr) {
            continue;
        }

        p_value->resize(p_data_i[0]);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;

        while(currOffset < p_data_i[0]) {
            SubIoRequest currSubIo;
            currSubIo.app_buff = (void*)p_value->data() + currOffset;
            currSubIo.real_size = (p_data_i[0] - currOffset) < PageSize ? (p_data_i[0] - currOffset) : PageSize;
            currSubIo.offset = p_data_i[dataIdx] * PageSize;
            currSubIo.posting_id = i;
            subIoRequests.push_back(currSubIo);
            subIoRequestCount[i]++;
            read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
        }
    }

    multi_read_times[id]++;

    for (int currSubIoStartId = 0; currSubIoStartId < subIoRequests.size(); currSubIoStartId += batch_size) {
        iocbs.clear();
        int currSubIoEndId = (currSubIoStartId + batch_size) > subIoRequests.size() ? subIoRequests.size() : currSubIoStartId + batch_size;
        int currSubIoIdx = currSubIoStartId;
        int totalToSubmit = currSubIoEndId - currSubIoStartId;
        int totalSubmitted = 0, totalDone = 0;
        for (int i = 0; i < totalToSubmit; i++) {
            auto currSubIoIdx = currSubIoStartId + i;
            if (!m_pWriteBuffer->get(subIoRequests[currSubIoIdx].offset, subIoRequests[currSubIoIdx].app_buff, subIoRequests[currSubIoIdx].real_size)) {
                auto currSubIo = m_currIoContext.free_sub_io_requests.front();
                m_currIoContext.free_sub_io_requests.pop();
                currSubIo->app_buff = subIoRequests[currSubIoIdx].app_buff;
                currSubIo->real_size = subIoRequests[currSubIoIdx].real_size;
                currSubIo->posting_id = subIoRequests[currSubIoIdx].posting_id;
                currSubIo->myiocb.aio_lio_opcode = 0; // IO_CMD_PREAD
                currSubIo->myiocb.aio_offset = subIoRequests[currSubIoIdx].offset;
                currSubIo->myiocb.aio_nbytes = subIoRequests[currSubIoIdx].real_size;
                iocbs.push_back(&(currSubIo->myiocb));
            }
            else {
                subIoRequestCount[subIoRequests[currSubIoIdx].posting_id]--;
                read_complete_vec[id]++;
            }
            currSubIoIdx++;
        }
        totalToSubmit = iocbs.size();
        auto begin = std::chrono::high_resolution_clock::now();
        int ret = dfs_multi_pread(cid, fd, totalToSubmit, iocbs.data());
        auto end = std::chrono::high_resolution_clock::now();
        multi_read_time_vec[id] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        if (ret >= 0) {
            for (int i = 0; i < totalToSubmit; i++) {
                auto currSubIo = reinterpret_cast<SubIoRequest*>(iocbs[i]->aio_data);
                subIoRequestCount[currSubIo->posting_id]--;
                m_currIoContext.free_sub_io_requests.push(currSubIo);
                memcpy(currSubIo->app_buff, currSubIo->myiocb.aio_buf, currSubIo->real_size);
                read_complete_vec[id]++;
            }
        }
        else {
            for (int i = 0; i < totalToSubmit; i++) {
                auto currSubIo = reinterpret_cast<SubIoRequest*>(iocbs[i]->aio_data);
                m_currIoContext.free_sub_io_requests.push(currSubIo);
            }
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            break;
        }
    }

    bool is_timeout = false;
    for (int i = 0; i < subIoRequestCount.size(); i++) {
        if (subIoRequestCount[i] != 0) {
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks (batch) : timeout\n");
            (*p_values)[i].clear();
            is_timeout = true;
        }
    }
    if (is_timeout) {
        m_batchReadTimeouts++;
    }
    return true;
}


bool LeoFSIO::BlockController::ReadBlocksAsync(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout) {
    auto t1 = std::chrono::high_resolution_clock::now();
    m_batchReadTimes++;
    p_values->resize(p_data.size());
    const int batch_size = m_batchSize;
    std::vector<dfs_iocb*> iocbs(batch_size);
    std::vector<dfs_io_event> events(batch_size);
    std::vector<SubIoRequest> subIoRequests;
    std::vector<int> subIoRequestCount(p_data.size(), 0);
    subIoRequests.reserve(256);
    for(size_t i = 0; i < p_data.size(); i++) {
        AddressType* p_data_i = p_data[i];
        std::string* p_value = &((*p_values)[i]);

        if (p_data_i == nullptr) {
            continue;
        }

        p_value->resize(p_data_i[0]);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;

        while(currOffset < p_data_i[0]) {
            SubIoRequest currSubIo;
            currSubIo.app_buff = (void*)p_value->data() + currOffset;
            currSubIo.real_size = (p_data_i[0] - currOffset) < PageSize ? (p_data_i[0] - currOffset) : PageSize;
            currSubIo.offset = p_data_i[dataIdx] * PageSize;
            currSubIo.posting_id = i;
            subIoRequests.push_back(currSubIo);
            subIoRequestCount[i]++;
            read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
        }
    }

    // Clear timeout I/Os
    while (m_currIoContext.free_sub_io_requests.size() < m_ssdLeoFSDepth) {
        int wait = m_ssdLeoFSDepth - m_currIoContext.free_sub_io_requests.size();
        std::vector<dfs_io_event> events(wait);
        dfs_timespec timeout_ts {0, 0};
        auto d = dfs_io_getevents(cid, iocp, wait, wait, events.data(), &timeout_ts);
        for (int i = 0; i < d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
    }

    dfs_timespec timeout_ts {0, timeout.count() * 1000};
    
    for (int currSubIoStartId = 0; currSubIoStartId < subIoRequests.size(); currSubIoStartId += batch_size) {
        int currSubIoEndId = (currSubIoStartId + batch_size) > subIoRequests.size() ? subIoRequests.size() : currSubIoStartId + batch_size;
        int currSubIoIdx = currSubIoStartId;
        int totalToSubmit = currSubIoEndId - currSubIoStartId;
        int totalSubmitted = 0, totalDone = 0;
        for (int i = 0; i < totalToSubmit; i++) {
            auto currSubIoIdx = currSubIoStartId + i;
            auto currSubIo = m_currIoContext.free_sub_io_requests.front();
            m_currIoContext.free_sub_io_requests.pop();
            currSubIo->app_buff = subIoRequests[currSubIoIdx].app_buff;
            currSubIo->real_size = subIoRequests[currSubIoIdx].real_size;
            currSubIo->posting_id = subIoRequests[currSubIoIdx].posting_id;
            currSubIo->myiocb.aio_lio_opcode = 0; // IO_CMD_PREAD
            currSubIo->myiocb.aio_offset = subIoRequests[currSubIoIdx].offset;
            currSubIo->myiocb.aio_nbytes = PageSize;
            iocbs[i] = &(currSubIo->myiocb);
            currSubIoIdx++;
        }
        while (totalDone < totalToSubmit) {
            // Submit all I/Os
            if(totalSubmitted < totalToSubmit) {
                int s = dfs_io_submit(cid, iocp, totalToSubmit - totalSubmitted, iocbs.data() + totalSubmitted);
                if(s > 0) {
                    totalSubmitted += s;
                }
            }
            int wait = totalSubmitted - totalDone;
            auto d = dfs_io_getevents(cid, iocp, wait, wait, events.data() + totalDone, &timeout_ts);
            for (int i = totalDone; i < totalDone + d; i++) {
                auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
                memcpy(req->app_buff, req->myiocb.aio_buf, req->real_size);
                subIoRequestCount[req->posting_id]--;
                req->app_buff = nullptr;
                m_currIoContext.free_sub_io_requests.push(req);
            }
            totalDone += d;
            read_complete_vec[id] += d;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            break;
        }
    }

    bool is_timeout = false;
    for (int i = 0; i < subIoRequestCount.size(); i++) {
        if (subIoRequestCount[i] != 0) {
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks (batch) : timeout\n");
            (*p_values)[i].clear();
            is_timeout = true;
        }
    }
    if (is_timeout) {
        m_batchReadTimeouts++;
    }
    return true;
}

bool LeoFSIO::BlockController::WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 5";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::WriteBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    AddressType currBlockIdx = 0;
    int totalSize = p_value.size();

    // Submit all I/Os
    write_submit_vec[id] += p_size;
    for (int i = 0; i  < p_size; i++) {
        void *buf = (void*)p_value.data() + currBlockIdx * PageSize;
        uint64_t real_size = (PageSize * (currBlockIdx + 1)) > totalSize ? (totalSize - currBlockIdx * PageSize) : PageSize;
        uint64_t offset = p_data[currBlockIdx] * PageSize;
        if (offset < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::WriteBlocks: offset is negative\n");
            exit(1);
        }
        memset(aligned_buf, 0, PageSize);
        memcpy(aligned_buf, buf, real_size);
        auto ret = dfs_pwrite(cid, fd, aligned_buf, real_size, offset);
        if (ret < real_size) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::WriteBlocks: dfs_pwrite failed\n");
            return false;
        }
        // dfs_fsync(cid, fd);
        currBlockIdx++;
        write_complete_vec[id]++;
    }
    
    return true;
}

bool LeoFSIO::BlockController::NewWriteBlocks(AddressType* p_data, int p_size, const std::string& p_value) {
    AddressType currBlockIdx = 0;
    if (p_size == 0) {
        return true;
    }
    int totalSize = p_value.size();
    std::vector<dfs_iocb*> iocbs(p_size);

    // Submit all I/Os
    write_submit_vec[id] += p_size;
    for (int i = 0; i  < p_size; i++) {
        auto currSubIo = m_currIoContext.free_sub_io_requests.front();
        m_currIoContext.free_sub_io_requests.pop();
        iocbs[i] = &(currSubIo->myiocb);
        void *buf = (void*)p_value.data() + currBlockIdx * PageSize;
        uint64_t real_size = (PageSize * (currBlockIdx + 1)) > totalSize ? (totalSize - currBlockIdx * PageSize) : PageSize;
        uint64_t offset = p_data[currBlockIdx] * PageSize;
        if (offset < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::WriteBlocks: offset is negative\n");
            exit(1);
        }
        // memset(aligned_buf, 0, PageSize);
        memcpy(iocbs[i]->aio_buf, buf, real_size);
        iocbs[i]->aio_offset = offset;
        iocbs[i]->aio_nbytes = real_size;
        currBlockIdx++;
        // write_complete_vec[id]++;
    }

    auto ret = dfs_multi_pwrite(cid, fd, p_size, iocbs.data());

    for (int i = 0; i < p_size; i++) {
        auto req = reinterpret_cast<SubIoRequest*>(iocbs[i]->aio_data);
        m_currIoContext.free_sub_io_requests.push(req);
    }

    if (ret < p_size) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::WriteBlocks: dfs_pwrite failed\n");
        return false;
    }
    
    write_complete_vec[id] += p_size;
    
    return true;
}

bool LeoFSIO::BlockController::BufferedWriteBlocks(AddressType* p_data, int p_size, const std::string& p_value) {
    AddressType currBlockIdx = 0;
    if (p_size == 0) {
        return true;
    }
    int totalSize = p_value.size();

    // Submit all I/Os
    write_submit_vec[id] += p_size;
    for (int i = 0; i  < p_size; i++) {
        void *buf = (void*)p_value.data() + currBlockIdx * PageSize;
        uint64_t real_size = (PageSize * (currBlockIdx + 1)) > totalSize ? (totalSize - currBlockIdx * PageSize) : PageSize;
        uint64_t offset = p_data[currBlockIdx] * PageSize;
        if (offset < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LeoFSIO::BlockController::WriteBlocks: offset is negative\n");
            exit(1);
        }
        m_pWriteBuffer->put(offset, buf, real_size);
        currBlockIdx++;
        // write_complete_vec[id]++;
    }
    
    write_complete_vec[id] += p_size;
    
    return true;
}

int64_t Sum(std::vector<int64_t>& vec) {
    int64_t sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}


bool LeoFSIO::BlockController::IOStatistics() {
    int currReadCount = 0;
    int read_submit_count = 0;
    int currWriteCount = 0;
    int write_submit_count = 0;
    int64_t read_blocks_time = 0;
    int64_t read_bytes_count = 0;
    int64_t write_bytes_count = 0;
    for (int i = 0; i < read_complete_vec.size(); i++) {
        currReadCount += read_complete_vec[i];
    }
    for (int i = 0; i < read_submit_vec.size(); i++) {
        read_submit_count += read_submit_vec[i];
    }
    for (int i = 0; i < write_complete_vec.size(); i++) {
        currWriteCount += write_complete_vec[i];
    }
    for (int i = 0; i < write_submit_vec.size(); i++) {
        write_submit_count += write_submit_vec[i];
    }
    for (int i = 0; i < read_bytes_vec.size(); i++) {
        read_bytes_count += read_bytes_vec[i];
    }
    for (int i = 0; i < write_bytes_vec.size(); i++) {
        write_bytes_count += write_bytes_vec[i];
    }
    for (int i = 0; i < read_blocks_time_vec.size(); i++) {
        read_blocks_time += read_blocks_time_vec[i];
    }
    
    int currIOCount = currReadCount + currWriteCount;
    int diffIOCount = currIOCount - m_preIOCompleteCount;
    m_preIOCompleteCount = currIOCount;

    int64_t currBytesCount = read_bytes_count + write_bytes_count;
    int64_t diffBytesCount = currBytesCount - m_preIOBytes;
    m_preIOBytes = currBytesCount;

    auto currTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currTime - m_preTime);
    m_preTime = currTime;

    double currIOPS = (double)diffIOCount * 1000 / duration.count();
    double currBandWidth = (double)diffIOCount * PageSize / 1024 * 1000 / 1024 * 1000 / duration.count();
    // double currBandWidth = (double)diffBytesCount / 1024 * 1000 / 1024 * 1000 / duration.count();

    std::cout << "Diff IO Count: " << diffIOCount << " Time: " << duration.count() << "us" << std::endl;
    std::cout << "IOPS: " << currIOPS << "k Bandwidth: " << currBandWidth << "MB/s" << std::endl;
    std::cout << "Read Count: " << currReadCount << " Write Count: " << currWriteCount << " Read Submit Count: " << read_submit_count << " Write Submit Count: " << write_submit_count << std::endl;
    std::cout << "Read Bytes Count: " << read_bytes_count << " Write Bytes Count: " << write_bytes_count << std::endl;
    std::cout << "Remain free IO requests: " << m_currIoContext.free_sub_io_requests.size() << std::endl;
    std::cout << "Read Blocks Time: " << read_blocks_time << "ns" << std::endl;
    std::cout << "Batch Read Times: " << m_batchReadTimes.load() << " Batch Read Timeouts: " << m_batchReadTimeouts.load() << std::endl;
    std::cout << "dfs_multi_pread avg time: " << Sum(multi_read_time_vec) / max(Sum(multi_read_times), 1L) << "ns" << std::endl;

    dfs_get_io_stats();
    if(m_pWriteBuffer) {
        m_pWriteBuffer->getStats();
    }
    return true;
}

bool LeoFSIO::BlockController::ShutDown() {
    std::lock_guard<std::mutex> lock(m_initMutex);
    SubIoRequest* currSubIo;
    m_numInitCalled--;
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LeoFSIO::BlockController::ShutDown\n");
    if (m_numInitCalled == 0) {
        m_LeoFSThreadExiting = true;
        pthread_join(m_LeoFSTid, NULL);
        while (!m_blockAddresses.empty()) {
            AddressType currBlockAddress;
            m_blockAddresses.try_pop(currBlockAddress);
        }
        while (!m_cidFds.empty()) {
            std::pair<int, int> currCidFd;
            int cid = -1, fd = -1;
            if (m_cidFds.try_pop(currCidFd)) {
                cid = currCidFd.first;
                fd = currCidFd.second;
                if (cid >= 0) {
                    if (fd >= 0) {
                        dfs_close(cid, fd);
                    }
                    dfs_disconnect(cid);
                }
            }
        }
        delete m_pWriteBuffer;
        // dfs_close(cid, fd);
        // dfs_disconnect(cid);
    }
    if (cid >= 0) {
        // dfs_dump_io_stats(cid);
        if (fd >= 0){
            m_cidFds.push({cid, fd});
        }
        else {
            dfs_disconnect(cid);
        }
    }
    
    m_idQueue.push(id);
    free(aligned_buf);
    dfs_io_destroy(cid, iocp);
    for (auto &sr : m_currIoContext.sub_io_requests) {
        sr.app_buff = nullptr;
        auto buf_ptr = sr.myiocb.aio_buf;
        free(buf_ptr);
        sr.myiocb.aio_buf = 0;
    }
    while(m_currIoContext.free_sub_io_requests.size()) {
        m_currIoContext.free_sub_io_requests.pop();
    }
    return true;
}

} // namespace SPTAG::SPANN
