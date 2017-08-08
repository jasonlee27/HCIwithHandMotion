// emptyDialog.cpp : implementation file
//

#include "stdafx.h"
#include "MFC_DVProject_04272016_ver2013.h"
#include "emptyDialog.h"
#include "afxdialogex.h"


// emptyDialog dialog

IMPLEMENT_DYNAMIC(emptyDialog, CDialogEx)

emptyDialog::emptyDialog(CWnd* pParent /*=NULL*/)
	: CDialogEx(emptyDialog::IDD, pParent)
{

}

emptyDialog::~emptyDialog()
{
}

void emptyDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(emptyDialog, CDialogEx)
END_MESSAGE_MAP()


// emptyDialog message handlers
