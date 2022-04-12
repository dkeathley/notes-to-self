;; Set the package installation directory so that packages aren't stored in the
;; ~/.emacs.d/elpa path.
(require 'package)
(setq package-user-dir (expand-file-name "./.packages"))
(setq package-archives '(("melpa" . "https://melpa.org/packages/")
                         ("elpa" . "https://elpa.gnu.org/packages/")))

;; Initialize the package system
(package-initialize)
(unless package-archive-contents
  (package-refresh-contents))

;; Install dependencies -- only run once, then comment out
;;(package-install 'htmlize)
;;(package-install 'org-cite)
;;(package-install 'org)
;;(package-install 'citeproc-org)
;;(package-install 'ob-ipython)

(require 'citeproc-org)

(citeproc-org-setup)

;; Load the publishing system
(require 'ox-publish)

(require 'org-id)
(setq org-id-link-to-org-use-id 'create-if-interactive-and-no-custom-id)

(setq citeproc-org-html-bib-header nil)

;;Org-Babel Ipython
(setq python-shell-interpreter "python3")
(org-babel-do-load-languages
 'org-babel-load-languages
 '((ipython . t)
   (octave . t)
   ;; other languages..
   ))
(setq org-src-fontify-natively t)
(setq org-src-preserve-indentation t)
(setq org-src-tab-acts-natively t)
(setq org-confirm-babel-evaluate nil)

;;Ensure inheritance for org-attach link handling
(require 'org-attach)
(setq org-attach-use-inheritance t)
(setq org-attach-dir-relative t)

(require 'oc-csl)

;; Define the publishing project
(setq org-publish-project-alist
      (list
       (list "org-site:main"
             :recursive t
             :base-directory "./"
             :publishing-function 'org-html-publish-to-html
             :publishing-directory "../public"
             :with-author nil           ;; Don't include author name
             :with-creator t            ;; Include Emacs and Org versions in footer
             :with-toc t                ;; Include a table of contents
             :section-numbers nil       ;; Don't include section numbers
             :time-stamp-file nil
			 :with-properties nil
			 :with-drawer t
			 :with-tags nil)

	   (list "org-static"
			 :base-directory "./"
			 :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf\\|mat\\|csv\\|py"
			 :publishing-directory "./public"
			 :recursive t
			 :publishing-function 'org-publish-attachment
			 )

	   )

	  )    ;; Don't include time stamp in file

;; Create separate class for drawers so that they can be formatted with titles.  
(defun my-org-export-format-drawer (name content)
  (concat "<div class=\"drawer-title\">" (capitalize name) "</div>\n"
		  "<div class=\"drawer-" (downcase name) "\">\n"
          content
          "\n</div>"))
(setq org-html-format-drawer-function 'my-org-export-format-drawer)


;;(setq org-html-validation-link nil)

;; Customize the HTML output
;; -- Note: When actually publishing, I could easily link to the web version of the .css file
;;          for styling.  This would always work from any page within the site.  
(setq org-html-validation-link nil            ;; Don't show validation link
      org-html-head-include-scripts nil       ;; Use our own scripts
      org-html-head-include-default-style nil ;; Use our own styles
	  org-html-head "<link rel=\"stylesheet\" href=\"static/style.css\" />"
	  )
	  ;;org-html-head "<link rel=\"stylesheet\" href=\"https://cdn.simplecss.org/simple.min.css\" />")

;; Generate the site output
(org-publish-all t)

(message "Build complete!")


