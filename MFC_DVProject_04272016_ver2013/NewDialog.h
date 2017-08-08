#pragma once


// NewDialog dialog

class NewDialog : public CDialogEx
{
	DECLARE_DYNAMIC(NewDialog)

public:
	NewDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~NewDialog();

// Dialog Data
	enum { IDD = IDD_MFC_DVPROJECT_04272016_VER2013_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
};
