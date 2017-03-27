

#include "stdafx.h"
#include "trace.h"



CTRACE::CTRACE(wchar_t *name): m_status(0), fp(0)
{
        fp = _wfopen(name, L"wt");
        if (!fp)
                m_status = -1;
}
CTRACE::~CTRACE()
{
        if (fp)
                fclose(fp);
}




void CTRACE::dump(float f)
{
        fwprintf(fp, L"%f\n", f);
}
void CTRACE::dump(int d)
{
        fwprintf(fp, L"%d\n", d);
}
void CTRACE::flush(void)
{
        fflush(fp);
}