// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_DFSIO_H_
#define _SPTAG_HELPER_DFSIO_H_

#include "leofs.h"
#include <cstdint>
#include <functional>
#include <fstream>
#include <string.h>
#include <memory>

namespace SPTAG
{
    namespace Helper
    {
        enum class DFSIOScenario
        {
            DIS_BulkRead = 0,
            DIS_UserRead,
            DIS_HighPriorityUserRead,
            DIS_BulkWrite,
            DIS_UserWrite,
            DIS_HighPriorityUserWrite,
            DIS_Count
        };

        struct AsyncReadRequest
        {
            std::uint64_t m_offset;
            std::uint64_t m_readSize;
            char* m_buffer;
            std::function<void(bool)> m_callback;
            int m_status;

            // Carry items like counter for callback to process.
            void* m_payload;
            bool m_success;

            // Carry exension metadata needed by some DFSIO implementations
            void* m_extension;

            AsyncReadRequest() : m_offset(0), m_readSize(0), m_buffer(nullptr), m_status(0), m_payload(nullptr), m_success(false), m_extension(nullptr) {}
        };

        class DFSIO
        {
        public:
            DFSIO(DFSIOScenario scenario = DFSIOScenario::DIS_UserRead) {}

            virtual ~DFSIO() {}

            virtual bool Initialize(int cid, const char* dfs_config, const char* filePath, int oflag, int openMode,
                // Max read/write buffer size.
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4) = 0;

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX) = 0;

            virtual std::uint64_t WriteBinary(std::uint64_t writeSize, const char* buffer, std::uint64_t offset = UINT64_MAX) = 0;

            virtual std::uint64_t ReadString(std::uint64_t& readSize, std::unique_ptr<char[]>& buffer, char delim = '\n', std::uint64_t offset = UINT64_MAX) = 0;

            virtual std::uint64_t WriteString(const char* buffer, std::uint64_t offset = UINT64_MAX) = 0;

            virtual bool ReadFileAsync(AsyncReadRequest& readRequest) { return false; }
            
            virtual bool BatchReadFile(AsyncReadRequest* readRequests, std::uint32_t requestCount) { return false; }

            virtual bool BatchCleanRequests(SPTAG::Helper::AsyncReadRequest* readRequests, std::uint32_t requestCount) { return false; }

            virtual std::uint64_t TellP() = 0;

            virtual void ShutDown() = 0; 
        };

        class SimpleLeoFSIO : public DFSIO
        {
        public:
            SimpleLeoFSIO(DFSIOScenario scenario = DFSIOScenario::DIS_UserRead) {}

            virtual ~SimpleLeoFSIO() { ShutDown(); }

            virtual bool Initialize(int m_cid, const char* dfs_config, const char* filePath, int oflag, int openMode,
                // Max read/write buffer size.
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4)
            {
                if (m_cid < 0 && dfs_config == nullptr) {
                    return false;
                }
                
                if (m_cid >= 0) {
                    cid = m_cid;
                    own_cid = false;
                } 
                else {
                    cid = dfs_connect_config(dfs_config);
                    own_cid = true;
                    if (cid < 0) {
                        return false;
                    }
                }

                fd = dfs_open(cid, filePath, oflag, openMode);
                if (fd < 0) {
                    return false;
                }
                return true;
            }

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) {
                    if (dfs_lseek(cid, fd, offset, SEEK_SET) < 0) {
                        return 0;
                    }
                }
                auto ret = dfs_read(cid, fd, buffer, readSize);
                if (ret < 0) {
                    return 0;
                }
                return (std::uint64_t)ret;
            }

            virtual std::uint64_t WriteBinary(std::uint64_t writeSize, const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) {
                    if (dfs_lseek(cid, fd, offset, SEEK_SET) < 0) {
                        return 0;
                    }
                }
                auto ret = dfs_write(cid, fd, buffer, writeSize);
                if (ret < 0) {
                    return 0;
                }
                return (std::uint64_t)ret;
            }

            virtual std::uint64_t ReadString(std::uint64_t& readSize, std::unique_ptr<char[]>& buffer, char delim = '\n', std::uint64_t offset = UINT64_MAX)
            {
                // TODO: 实现高效的ReadString
                return 0;
            }

            virtual std::uint64_t WriteString(const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                return WriteBinary(strlen(buffer), (const char*)buffer, offset);
            }

            virtual std::uint64_t TellP()
            {
                return dfs_lseek(cid, fd, 0, SEEK_CUR);
            }

            virtual void ShutDown()
            {
                dfs_close(cid, fd);
                if (own_cid) {
                    dfs_disconnect(cid);
                }
            }

        private:
            std::unique_ptr<std::fstream> m_handle;
            int cid;
            int fd;
            bool own_cid;
        };

        class SimpleLeoFSBufferIO : public DFSIO
        {
        public:
            struct streambuf : public std::basic_streambuf<char>
            {
                streambuf() {}

                streambuf(char* buffer, size_t size)
                {
                    setg(buffer, buffer, buffer + size);
                    setp(buffer, buffer + size);
                }

                std::uint64_t tellp()
                {
                    if (pptr()) return pptr() - pbase();
                    return 0;
                }
            };

            SimpleLeoFSBufferIO(DFSIOScenario scenario = DFSIOScenario::DIS_UserRead) {}

            virtual ~SimpleLeoFSBufferIO()
            {
                ShutDown();
            }

            virtual bool Initialize(const char* filePath, int openMode,
                // Max read/write buffer size.
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4)
            {
                if (filePath != nullptr)
                    m_handle.reset(new streambuf((char*)filePath, maxIOSize));
                else
                    m_handle.reset(new streambuf());
                return true;
            }

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->pubseekpos(offset);
                return m_handle->sgetn((char*)buffer, readSize);
            }

            virtual std::uint64_t WriteBinary(std::uint64_t writeSize, const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->pubseekpos(offset);
                if ((std::uint64_t)m_handle->sputn((const char*)buffer, writeSize) < writeSize) return 0;
                return writeSize;
            }

            virtual std::uint64_t ReadString(std::uint64_t& readSize, std::unique_ptr<char[]>& buffer, char delim = '\n', std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->pubseekpos(offset);
                std::uint64_t readCount = 0;
                for (int _Meta = m_handle->sgetc();; _Meta = m_handle->snextc()) {
                    if (_Meta == '\r') _Meta = '\n';

                    if (readCount >= readSize) { // buffer full
                        readSize *= 2;
                        std::unique_ptr<char[]> newBuffer(new char[readSize]);
                        memcpy(newBuffer.get(), buffer.get(), readCount);
                        buffer.swap(newBuffer);
                    }

                    if (_Meta == EOF) { // eof
                        buffer[readCount] = '\0';
                        break;
                    }
                    else if (_Meta == delim) { // got a delimiter, discard it and quit
                        buffer[readCount++] = '\0';
                        m_handle->sbumpc();
                        if (delim == '\n' && m_handle->sgetc() == '\n') {
                            readCount++;
                            m_handle->sbumpc();
                        }
                        break;
                    }
                    else { // got a character, add it to string
                        buffer[readCount++] = std::char_traits<char>::to_char_type(_Meta);
                    }
                }
                return readCount;
            }

            virtual std::uint64_t WriteString(const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                return WriteBinary(strlen(buffer), (const char*)buffer, offset);
            }

            virtual std::uint64_t TellP()
            { 
                return m_handle->tellp(); 
            }

            virtual void ShutDown() {}

        private:
            std::unique_ptr<streambuf> m_handle;
        };
    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_DFSIO_H_
