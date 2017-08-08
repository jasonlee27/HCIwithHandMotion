
// MFC_DVProject_04272016_ver2013Dlg.h : header file
//

#pragma once


// CMFC_DVProject_04272016_ver2013Dlg dialog
class CMFC_DVProject_04272016_ver2013Dlg : public CDialogEx
{
// Construction
public:
	CMFC_DVProject_04272016_ver2013Dlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_MFC_DVPROJECT_04272016_VER2013_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedButton4();
};
