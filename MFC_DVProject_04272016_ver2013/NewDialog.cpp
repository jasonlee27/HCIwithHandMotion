// NewDialog.cpp : implementation file
//

#include "stdafx.h"
#include "MFC_DVProject_04272016_ver2013.h"
#include "NewDialog.h"
#include "afxdialogex.h"


// NewDialog dialog

IMPLEMENT_DYNAMIC(NewDialog, CDialogEx)

NewDialog::NewDialog(CWnd* pParent /*=NULL*/)
	: CDialogEx(NewDialog::IDD, pParent)
{

}

NewDialog::~NewDialog()
{
}

void NewDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(NewDialog, CDialogEx)
END_MESSAGE_MAP()


// NewDialog message handlers
