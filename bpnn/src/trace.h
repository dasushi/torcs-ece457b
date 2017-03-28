

#pragma once


class CTRACE
{

        FILE *fp;   //file handle

public:
        int m_status;

        CTRACE(wchar_t *name);
        ~CTRACE();

        void dump(float f);
        void dump(int d);
        void flush(void);
};

