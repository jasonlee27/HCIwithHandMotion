
// MFC_DVProject_04272016_ver2013.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CMFC_DVProject_04272016_ver2013App:
// See MFC_DVProject_04272016_ver2013.cpp for the implementation of this class
//

class CMFC_DVProject_04272016_ver2013App : public CWinApp
{
public:
	CMFC_DVProject_04272016_ver2013App();

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CMFC_DVProject_04272016_ver2013App theApp;